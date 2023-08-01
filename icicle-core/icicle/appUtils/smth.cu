#include "../primitives/field.cuh"
#if CURVE == 12381
#include "../curves/bls12_381.cuh"
using namespace bls12;
// namespace bls12_381
#elif CURVE == 254
#include "../curves/bn254.cuh"
using namespace bn254;
// namespace bn_254
#endif

typedef Field<fq_config> point_field_t;

extern "C" int do_smth(point_field_t *arr)
{
    try
    {
#if CURVE == 12381
        std::cout << "from bls12_381: "
#elif CURVE == 254
        std::cout << "from bn254: "
#endif
        << arr[0] << std::endl;
        return point_field_t::TLC;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}
