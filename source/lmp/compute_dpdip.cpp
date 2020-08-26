#include <iostream>
#include <iomanip>
#include <limits>
#include <mpi.h>
#include "atom.h"
#include "domain.h"
#include "comm.h"
#include "force.h"
#include "update.h"
#include "error.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "fix.h"
#include "fix_dplr.h"
#include "pppm_dplr.h"
#include "compute_dipole.h"


using namespace LAMMPS_NS;
using namespace FixConst;
using namespace std;

ComputeDPDip::ComputeDPDip(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg != 3) error->all(FLERR,"Illegal compute dipole command");

  vector_flag = 1;
  size_vector = 3;
  extvector = 1;
  vector = new double[size_vector];
}

ComputeDPDip::~ComputeDPDip() {
  delete[] vector;
}

void ComputeDPDip::init()
{
}

void
FixDPLR::get_valid_pairs(vector<pair<int,int> >& pairs)
{
  pairs.clear();
  
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;
  vector<int > dtype (nall);
  // get type
  {
    int *type = atom->type;
    for (int ii = 0; ii < nall; ++ii){
      dtype[ii] = type[ii] - 1;
    }
  }

  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  for (int ii = 0; ii < nbondlist; ++ii){
    int idx0=-1, idx1=-1;
    int bd_type = bondlist[ii][2] - 1;
    if ( ! binary_search(bond_type.begin(), bond_type.end(), bd_type) ){
      continue;
    }
    if (binary_search(sel_type.begin(), sel_type.end(), dtype[bondlist[ii][0]]) 
  && 
  binary_search(dpl_type.begin(), dpl_type.end(), dtype[bondlist[ii][1]])
  ){
      idx0 = bondlist[ii][0];
      idx1 = bondlist[ii][1];
    }
    else if (binary_search(sel_type.begin(), sel_type.end(), dtype[bondlist[ii][1]])
       &&
       binary_search(dpl_type.begin(), dpl_type.end(), dtype[bondlist[ii][0]])
  ){
      idx0 = bondlist[ii][1];
      idx1 = bondlist[ii][0];
    }
    else {
      error->all(FLERR, "find a bonded pair the types of which are not associated");
    }
    if ( ! (idx0 < nlocal && idx1 < nlocal) ){
      error->all(FLERR, "find a bonded pair that is not on the same processor, something should not happen");
    }
    pairs.push_back(pair<int,int>(idx0, idx1));
  }
}

void ComputeDPDip::compute_vector()
{
  invoked_vector = update->ntimestep;

  double **x = atom->x;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;

  // declear inputs
  std::vector<int > dtype (nall);
  std::vector<FLOAT_PREC > dbox (9, 0) ;
  std::vector<FLOAT_PREC > dcoord (nall * 3, 0.);
  // get type
  for (int ii = 0; ii < nall; ++ii){
    dtype[ii] = type[ii] - 1;
  }  
  // get box
  dbox[0] = domain->h[0]; // xx
  dbox[4] = domain->h[1]; // yy
  dbox[8] = domain->h[2]; // zz
  dbox[7] = domain->h[3]; // zy
  dbox[6] = domain->h[4]; // zx
  dbox[3] = domain->h[5]; // yx
  // get coord
  for (int ii = 0; ii < nall; ++ii){
    for (int dd = 0; dd < 3; ++dd){
      dcoord[ii*3+dd] = x[ii][dd] - domain->boxlo[dd];
    }
  }
  // get lammps nlist
  NeighList * list = pair_nnp->list;
  LammpsNeighborList lmp_list (list->inum, list->ilist, list->numneigh, list->firstneigh);
  // declear output
  std::vector<FLOAT_PREC> tensor;
  // compute
  dpt.compute(tensor, dcoord, dtype, dbox, nghost, lmp_list);
  // selected type
  std::vector<int> dpl_type;
  for (int ii = 0; ii < sel_type.size(); ++ii){
    dpl_type.push_back(type_asso[sel_type[ii]]);
  }
  std::vector<int> sel_fwd, sel_bwd;
  int sel_nghost;
  select_by_type(sel_fwd, sel_bwd, sel_nghost, dcoord, dtype, nghost, sel_type);
  int sel_nall = sel_bwd.size();
  int sel_nloc = sel_nall - sel_nghost;
  std::vector<int> sel_type(sel_bwd.size());
  select_map<int>(sel_type, dtype, sel_fwd, 1);
  
  NNPAtomMap<FLOAT_PREC> nnp_map(sel_type.begin(), sel_type.begin() + sel_nloc);
  const std::vector<int> & sort_fwd_map(nnp_map.get_fwd_map());

  std::vector<pair<int,int> > valid_pairs;
  get_valid_pairs(valid_pairs);  
  
  std::vector<double > m_wfc (3, 0.0);
  int odim = dpt.output_dim();
  assert(odim == 3);
  for (int ii = 0; ii < valid_pairs.size(); ++ii){
    int idx0 = valid_pairs[ii].first;
    int idx1 = valid_pairs[ii].second;
    assert(idx0 < sel_fwd.size() && sel_fwd[idx0] < sort_fwd_map.size());
    int res_idx = sort_fwd_map[sel_fwd[idx0]];
    // int ret_idx = dpl_bwd[res_idx];
    for (int dd = 0; dd < 3; ++dd){
      x[idx1][dd] = x[idx0][dd] + tensor[res_idx * 3 + dd];
      m_wfc[dd] += x[idx1][dd];
    }
  }

  std::vector<double > m_o (3, 0.0);
  std::vector<double > m_h (3, 0.0);

  for (int ii = 0; ii < nlocal; ++ii)
  {
    for (int dd = 0; dd < 3; ++dd)
    {
      if (!type[ii]){
        m_o[dd] += dcoord[ii*3+dd];
      }
      else if (type[ii]){
        m_h[dd] += dcoord[ii*3+dd];
      }
    }
  }

  std::vector<double > dip (3, 0.0);
  for (int dd = 0; dd < 3; ++dd)
  {
    dip[dd] = (-2) * m_o[dd] + m_h[dd] + (-2) * m_wfc[dd]
  }  

  MPI_Allreduce(&dip, vector, 3, MPI_DOUBLE, MPI_SUM, world);
}
