#if defined(BLS12)
#include "../curves/bls12_381.cuh"
#endif
#if defined(BN254)
#include "../curves/bn254.cuh"
#endif
#include "../primitives/field.cuh"

typedef Field<fq_config> point_field_t;

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

extern "C" int do_smth1(point_field_t *arr)
{
    try
    {
        std::cout << arr[0] << std::endl;
        return point_field_t::TLC;
    }
    catch (const std::runtime_error &ex)
    {
        printf("error %s", ex.what());
        return -1;        
    }
}