#ifdef COMPUTE_CLASS

ComputeStyle(dpdip,ComputeDPDip)

#else

#ifndef LMP_COMPUTE_DPDIP_H
#define LMP_COMPUTE_DPDIP_H

#include <stdio.h>
#include "fix.h"
#include "compute.h"
#include "pair_nnp.h"
#include "DeepTensor.h"
#include "DataModifier.h"

#ifdef HIGH_PREC
#define FLOAT_PREC double
#else
#define FLOAT_PREC float
#endif

namespace LAMMPS_NS {
  class omputeDPDip : public Compute {
public:
   ComputeDPDip(class LAMMPS *, int, char **);
   virtual ~ComputeDPDip();

   int setmask();
   void init();
   void pre_force(int);
   virtual void compute_vector();
private:
    PairNNP * pair_nnp;
    DeepTensor dpt;
    DataModifier dtm;
    vector<int > sel_type;
    vector<int > dpl_type;
    vector<int > bond_type;
    map<int,int > type_asso;
    void get_valid_pairs(vector<pair<int,int> >& pairs);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

*/
