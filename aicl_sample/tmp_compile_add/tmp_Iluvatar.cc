#include "aicl.h"
#include <vector>
#include <memory>
#include <dmlc/logging.h>
#include <tvm/te/operation.h>
using namespace std;



aiclTensorDesc *aiclCreateTensorDesc(aiclDataType dataType, int numDims,
                                     const int64_t *dims,  aiclFormat format) {
    // std::unique_ptr<DLTensor> from ;
    DLTensor *from = new DLTensor();
    DLDataType dtype;
    switch (dataType) {
    case 0:   // AICL_FLOAT
        dtype = {kDLFloat, 32, 1};
        break;
    case 1:    // AICL_FLOAT16
        dtype = {kDLFloat, 16, 1};
        break;
    case 2:     // AICL_INT8
        dtype = {kDLInt, 8, 1};
        break;
    case 3:     // AICL_INT32
        dtype = {kDLInt, 32, 1};
        break;
    case 4:     // AICL_UINT8
        dtype = {kDLUInt, 8, 1};
        break;
    case 6:     // AICL_INT16
        dtype = {kDLInt, 16, 1};
        break;
    case 7:     // AICL_UINT16
        dtype = {kDLUInt, 16, 1};
        break;
    case 8:    // AICL_UINT32
        dtype = {kDLUInt, 32, 1};
        break;
    default:
        break;
    }
    from->dtype = dtype;
    from->ndim = static_cast<int>(numDims);
    from->shape = const_cast<int64_t *>(dims);
    // return (aiclTensorDesc *)from.get();
    return (aiclTensorDesc *)from;
}

size_t aiclGetTensorDescSize(const aiclTensorDesc *desc) {
    DLTensor *TDesc = const_cast<DLTensor*>((DLTensor*)desc);
    return tvm::runtime::GetDataSize(*TDesc);
}

size_t aiclGetTensorDescNumDims(const aiclTensorDesc *desc) {
    DLTensor *TDesc = const_cast<DLTensor*>((DLTensor*)desc);
    return TDesc->ndim;
}

int64_t aiclGetTensorDescDim(const aiclTensorDesc *desc, size_t index) {
    DLTensor *TDesc = const_cast<DLTensor*>((DLTensor*)desc);
    return TDesc->shape[index];
}

size_t aiclGetTensorDescElementCount(const aiclTensorDesc *desc) {
    DLTensor *TDesc = const_cast<DLTensor*>((DLTensor*)desc);
    size_t count = 1;
    for (int n = 0; n < TDesc->ndim; n++)
    {
        count *= TDesc->shape[n];
    }
    return count;
}

aiclDataType aiclGetTensorDescType(const aiclTensorDesc *desc) {
    DLTensor *TDesc = const_cast<DLTensor*>((DLTensor*)desc);
    // LOG(INFO) << (int)TDesc->dtype.code;
    // LOG(INFO) << (int)TDesc->dtype.bits;
    // LOG(INFO) << TDesc->dtype.lanes;
    switch ((int)TDesc->dtype.code)
    {
    case 0:
        if((int)TDesc->dtype.bits == 8){
            return AICL_INT8;
        } else if ((int)TDesc->dtype.bits == 16){
            return AICL_INT16;
        } else {
            return AICL_INT32;
        }
        break;
    case 1:
        if((int)TDesc->dtype.bits == 8){
            return AICL_UINT8;
        } else if ((int)TDesc->dtype.bits == 16){
            return AICL_UINT16;
        } else {
            return AICL_UINT32;
        }
        break;
    case 2:
        if((int)TDesc->dtype.bits == 16){
            return AICL_FLOAT16;
        } else {
            return AICL_FLOAT;
        }
        break;
    case 4:
        return AICL_BF16;
        break;
    default:
        return AICL_DT_UNDEFINED;
        break;
    }
}



int main(){
    aiclDataType datype = AICL_FLOAT16;
    int numD = 4;
    const int64_t dims[4] = {4, 5, 6, 7};
    aiclFormat form = AICL_FORMAT_ND;
    aiclTensorDesc *tmp = aiclCreateTensorDesc(datype, numD, dims, form);
    // size_t tmp_size = tvm::runtime::GetDataSize((DLTensor *)tmp);
    DLTensor *fromA = (DLTensor *)tmp;
    LOG(INFO) << "fromA->shape = " << fromA->shape;
    LOG(INFO) << "fromA->shape[0] = " << fromA->shape[0];
    LOG(INFO) << "fromA->shape[1] = " << fromA->shape[1];
    LOG(INFO) << "fromA->shape[2] = " << fromA->shape[2];
    LOG(INFO) << "fromA->shape[3] = " << fromA->shape[3];
    LOG(INFO) << "fromA->data = " << fromA->data;
    LOG(INFO) << "fromA->ndim = " << fromA->ndim;
    LOG(INFO) << "fromA->dtype.bits = " << (int)(fromA->dtype.bits);
    LOG(INFO) << "fromA->dtype.lanes = " << fromA->dtype.lanes;

    LOG(INFO) << "size = " << tvm::runtime::GetDataSize(*fromA);

    size_t tmp_size = tvm::runtime::GetDataSize(*fromA);
    LOG(INFO) << tmp_size;
    cout << tmp_size << endl;
    size_t t_size = aiclGetTensorDescSize(tmp);
    LOG(INFO) << t_size;
    size_t T_num = aiclGetTensorDescNumDims(tmp);
    LOG(INFO) << T_num;
    int num0 = 0;
    LOG(INFO) << aiclGetTensorDescDim(tmp, num0);
    int num1 = 1;
    LOG(INFO) << aiclGetTensorDescDim(tmp, num1);
    int num2 = 2;
    LOG(INFO) << aiclGetTensorDescDim(tmp, num2);
    int num3 = 3;
    LOG(INFO) << aiclGetTensorDescDim(tmp, num3);
    LOG(INFO) << aiclGetTensorDescElementCount(tmp);

    LOG(INFO) << "aiclDataType = " << aiclGetTensorDescType(tmp);

    return 0;
}