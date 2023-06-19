#include "../primitives/field.cuh"
#if defined(BLS12)
#include "../curves/bls12_381.cuh"
using namespace bls12;
#endif
#if defined(BN254)
#include "../curves/bn254.cuh"
using namespace bn254;
#endif
typedef Field<fq_config> point_field_t;

#if defined(BN254)
extern "C" int do_smth(point_field_t *arr)
{
    try
    {
        #if defined(BLS12)
        std::cout << "381" << std::endl;
        #endif
        #if defined(BN254)
        std::cout << "254" << std::endl;
        #endif
        std::cout << arr[0] << std::endl;
        return point_field_t::TLC;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}
#endif

#if defined(BLS12)
extern "C" int do_smth1(point_field_t *arr)
{
    try
    {
        #if defined(BLS12)
        std::cout << "381" << std::endl;
        #endif
        #if defined(BN254)
        std::cout << "254" << std::endl;
        #endif
        std::cout << arr[0] << std::endl;
        return point_field_t::TLC;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;
    }
}
#endif
