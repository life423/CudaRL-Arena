// A separate TU also prevents the per-link search being cloned at both fin calls.
#ifndef CUDARL_POND_GEOMETRY_RDC
#error "Compile the shared pond link search with CUDARL_POND_GEOMETRY_RDC"
#endif
#define CUDARL_DEFINE_POND_INSPECT_LINK
#define CUDARL_DEFINE_POND_SAMPLE
#ifdef CUDARL_POND_GEOMETRY_MODEL_NAMESPACE
#define cudarl CUDARL_POND_GEOMETRY_MODEL_NAMESPACE
#endif
#include "cuda/pond_math_profile.h"
