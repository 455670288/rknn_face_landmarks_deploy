/**
  * @ClassName yolo_image
  * @Description inference code for yolo
  * @Author raul.rao
  * @Date 2022/5/23 11:10
  * @Version 1.0
  */

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <ctime>

#include <cstdint>

#include "rknn_api.h"

#include "yolo_image.h"
#include "rga/rga.h"
#include "rga/im2d.h"
#include "rga/im2d_version.h"
#include "post_process.h"


#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

//#define DEBUG_DUMP
//#define EVAL_TIME
#define ZERO_COPY 1
#define DO_NOT_FLIP -1

int g_inf_count = 0;

int g_post_count = 0;

rknn_context ctx = 0;

bool created = false;

int img_width = 0;    // the width of the actual input image
int img_height = 0;   // the height of the actual input image

int m_in_width = 0;   // the width of the RKNN model input
int m_in_height = 0;  // the height of the RKNN model input
int m_in_channel = 0; // the channel of the RKNN model input

float scale_w = 0.0;
float scale_h = 0.0;

uint32_t n_input = 1;
uint32_t n_output = 3;

rknn_tensor_attr input_attrs[1];
rknn_tensor_attr output_attrs[1];

rknn_tensor_mem *input_mems[1];
rknn_tensor_mem *output_mems[1];

rga_buffer_t g_rga_src;
rga_buffer_t g_rga_dst;
unsigned char *model;
rknn_input_output_num io_num;

std::vector<float> out_scales;
std::vector<int32_t> out_zps;

double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }
//static unsigned char* load_model(const char* filename, int* model_size)
//{
//    FILE* fp = fopen(filename, "rb");
//    if (fp == nullptr) {
//        printf("fopen %s fail!\n", filename);
//        return NULL;
//    }
//    fseek(fp, 0, SEEK_END);
//    int model_len = ftell(fp);
//    unsigned char* model     = (unsigned char*)malloc(model_len);
//    fseek(fp, 0, SEEK_SET);
//    if (model_len != fread(model, 1, model_len, fp)) {
//        printf("fread %s fail!\n", filename);
//        free(model);
//        return NULL;
//    }
//    *model_size = model_len;
//    if (fp) {
//        fclose(fp);
//    }
//    return model;
//}
//
//
//static int rknn_GetTop(float* pfProb, float* pfMaxProb, uint32_t* pMaxClass, uint32_t outputCount, uint32_t topNum)
//{
//    uint32_t i, j;
//
//#define MAX_TOP_NUM 20
//    if (topNum > 20)
//        return 0;
//
//    memset(pfMaxProb, 0, sizeof(float) * topNum);
//    memset(pMaxClass, 0xff, sizeof(float) * topNum);
//
//    for (j = 0; j < topNum; j++) {
//        for (i = 0; i < outputCount; i++) {
//            if ((i == *(pMaxClass + 0)) || (i == *(pMaxClass + 1)) || (i == *(pMaxClass + 2)) || (i == *(pMaxClass + 3)) ||
//                (i == *(pMaxClass + 4))) {
//                continue;
//            }
//
//            if (pfProb[i] > *(pfMaxProb + j)) {
//                *(pfMaxProb + j) = pfProb[i];
//                *(pMaxClass + j) = i;
//            }
//        }
//    }
//
//    return 1;
//}
//
//
//int create(int im_height, int im_width, int im_channel, char *model_path)
//{
//    const int MODEL_IN_WIDTH = im_width;
//    const int MODEL_IN_HEIGHT = im_height;
//    const int MODEL_IN_CHANNELS = im_channel;
//
////    rknn_context ctx = 0;
//    int ret;
//    int model_len = 0;
////    unsigned char *model;
//
//    const char *img_path = "./assets/cat_224x224.jpg";
//
//    // Load image
//    cv::Mat orig_img = imread(img_path, cv::IMREAD_COLOR);
//    if (!orig_img.data) {
//        printf("cv::imread %s fail!\n", img_path);
//        return -1;
//    }
//
//    cv::Mat orig_img_rgb;
//    cv::cvtColor(orig_img, orig_img_rgb, cv::COLOR_BGR2RGB);
//
//    cv::Mat img = orig_img_rgb.clone();
//    if (orig_img.cols != MODEL_IN_WIDTH || orig_img.rows != MODEL_IN_HEIGHT) {
//        printf("resize %d %d to %d %d\n", orig_img.cols, orig_img.rows, MODEL_IN_WIDTH, MODEL_IN_HEIGHT);
//        cv::resize(orig_img_rgb, img, cv::Size(MODEL_IN_WIDTH, MODEL_IN_HEIGHT), 0, 0, cv::INTER_LINEAR);
//    }
//
//    FILE *fp = fopen(model_path, "rb");
//    if (fp == NULL) {
//        LOGE("fopen %s fail!\n", model_path);
//        return -1;
//    }
//    fseek(fp, 0, SEEK_END);
//    model_len = ftell(fp);
//    void *model = malloc(model_len);
//    fseek(fp, 0, SEEK_SET);
//    if (model_len != fread(model, 1, model_len, fp)) {
//        LOGE("fread %s fail!\n", model_path);
//        free(model);
//        fclose(fp);
//        return -1;
//    }
//    fclose(fp);
//
//    ret = rknn_init(&ctx, model, model_len, 0, NULL);
//    free(model);
//    if (ret < 0) {
//        printf("rknn_init fail! ret=%d\n", ret);
//        return -1;
//    }
//
//    // Get Model Input Output Info
////    rknn_input_output_num io_num;
//    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
//    if (ret != RKNN_SUCC) {
//        printf("rknn_query fail! ret=%d\n", ret);
//        return -1;
//    }
//    printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);
//
//    printf("input tensors:\n");
//    rknn_tensor_attr input_attrs[io_num.n_input];
//    memset(input_attrs, 0, sizeof(input_attrs));
//    for (int i = 0; i < io_num.n_input; i++) {
//        input_attrs[i].index = i;
//        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
//        if (ret != RKNN_SUCC) {
//            printf("rknn_query fail! ret=%d\n", ret);
//            return -1;
//        }
//
//    }
//
//    printf("output tensors:\n");
//    rknn_tensor_attr output_attrs[io_num.n_output];
//    memset(output_attrs, 0, sizeof(output_attrs));
//    for (int i = 0; i < io_num.n_output; i++) {
//        output_attrs[i].index = i;
//        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
//        if (ret != RKNN_SUCC) {
//            printf("rknn_query fail! ret=%d\n", ret);
//            return -1;
//        }
//
//    }
//
//    // Set Input Data
//    rknn_input inputs[1];
//    memset(inputs, 0, sizeof(inputs));
//    inputs[0].index = 0;
//    inputs[0].type = RKNN_TENSOR_UINT8;
//    inputs[0].size = img.cols * img.rows * img.channels() * sizeof(uint8_t);
//    inputs[0].fmt = RKNN_TENSOR_NHWC;
//    inputs[0].buf = img.data;
//
//    ret = rknn_inputs_set(ctx, io_num.n_input, inputs);
//    if (ret < 0) {
//        printf("rknn_input_set fail! ret=%d\n", ret);
//        return -1;
//    }
//    return 0;
//}

int create(int im_height, int im_width, int im_channel, char *model_path)
{
    img_height = im_height;
    img_width = im_width;

    LOGI("try rknn_init!")

    // 0. RGA version check
    LOGI("RGA API Version: %s", RGA_API_VERSION)
    // Please refer to the link to confirm the RGA driver version, make sure it is higher than 1.2.4
    // https://github.com/airockchip/librga/blob/main/docs/Rockchip_FAQ_RGA_CN.md#rga-driver

    // 1. Load model
    FILE *fp = fopen(model_path, "rb");
    if(fp == NULL) {
        LOGE("fopen %s fail!\n", model_path);
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    uint32_t model_len = ftell(fp);
    void *model = malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if(model_len != fread(model, 1, model_len, fp)) {
        LOGE("fread %s fail!\n", model_path);
        free(model);
        fclose(fp);
        return -1;
    }

    fclose(fp);

    // 2. Init RKNN model
    int ret = rknn_init(&ctx, model, model_len, 0, nullptr);
    free(model);

    if(ret < 0) {
        LOGE("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // 3. Query input/output attr.
    rknn_input_output_num io_num;
    rknn_query_cmd cmd = RKNN_QUERY_IN_OUT_NUM;
    // 3.1 Query input/output num.
    ret = rknn_query(ctx, cmd, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        LOGE("rknn_query io_num fail!ret=%d\n", ret);
        return -1;
    }
    n_input = io_num.n_input;
    n_output = io_num.n_output;

    // 3.2 Query input attributes
    memset(input_attrs, 0, n_input * sizeof(rknn_tensor_attr));
    for (int i = 0; i < n_input; ++i) {
        input_attrs[i].index = i;
        cmd = RKNN_QUERY_INPUT_ATTR;
        ret = rknn_query(ctx, cmd, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            LOGE("rknn_query input_attrs[%d] fail!ret=%d\n", i, ret);
            return -1;
        }
    }
    // 3.2.0 Update global model input shape.
    if (RKNN_TENSOR_NHWC == input_attrs[0].fmt) {
        m_in_height = input_attrs[0].dims[1];
        m_in_width = input_attrs[0].dims[2];
        m_in_channel = input_attrs[0].dims[3];
    } else if (RKNN_TENSOR_NCHW == input_attrs[0].fmt) {
        m_in_height = input_attrs[0].dims[2];
        m_in_width = input_attrs[0].dims[3];
        m_in_channel = input_attrs[0].dims[1];
    } else {
        LOGE("Unsupported model input layout: %d!\n", input_attrs[0].fmt);
        return -1;
    }

    // set scale_w, scale_h for post process
    scale_w = (float)m_in_width / img_width;
    scale_h = (float)m_in_height / img_height;
    LOGI("m_in_width = %d", m_in_width)
    LOGI("img_width = %d", img_width)
    LOGI("scale_w = %f", scale_w)
    LOGI("m_in_height = %d", m_in_height)
    LOGI("img_height = %d", img_height)
    LOGI("scale_h = %f", scale_h)

    // 3.3 Query output attributes
    memset(output_attrs, 0, n_output * sizeof(rknn_tensor_attr));
    for (int i = 0; i < n_output; ++i) {
        output_attrs[i].index = i;
        cmd = RKNN_QUERY_OUTPUT_ATTR;
        ret = rknn_query(ctx, cmd, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            LOGE("rknn_query output_attrs[%d] fail!ret=%d\n", i, ret);
            return -1;
        }
        // set out_scales/out_zps for post_process
        out_scales.push_back(output_attrs[i].scale);
        out_zps.push_back(output_attrs[i].zp);
    }

#if ZERO_COPY
    // 4. Set input/output buffer
    // 4.1 Set inputs memory
    // 4.1.1 Create input tensor memory, input data type is INT8, yolo has only 1 input.
    input_mems[0] = rknn_create_mem(ctx, input_attrs[0].size_with_stride * sizeof(char));
    memset(input_mems[0]->virt_addr, 0, input_attrs[0].size_with_stride * sizeof(char));
    // 4.1.2 Update input attrs
    input_attrs[0].index = 0;
    input_attrs[0].type = RKNN_TENSOR_UINT8;
    input_attrs[0].size = m_in_height * m_in_width * m_in_channel * sizeof(char);
    input_attrs[0].fmt = RKNN_TENSOR_NHWC;
    LOGI("input_attrs[0].size = %d", input_attrs[0].size)
    LOGI("m_in_height = %d", m_in_height)
    LOGI("m_in_width = %d", m_in_width)
    LOGI("m_in_channel = %d", m_in_channel)
    // TODO -- The efficiency of pass through will be higher, we need adjust the layout of input to
    //         meet the use condition of pass through.
    input_attrs[0].pass_through = 0;
    // 4.1.3 Set input buffer
    rknn_set_io_mem(ctx, input_mems[0], &(input_attrs[0]));
    // 4.1.4 bind virtual address to rga virtual address
    g_rga_dst = wrapbuffer_virtualaddr((void *)input_mems[0]->virt_addr, m_in_width, m_in_height,
                                       RK_FORMAT_RGB_888);

    LOGI("output_attrs[0].type = %d", output_attrs[0].type)
    // 4.2 Set outputs memory
    for (int i = 0; i < n_output; ++i) {
        // 4.2.1 Create output tensor memory, output data type is int8, post_process need int8 data.
        output_mems[i] = rknn_create_mem(ctx, output_attrs[i].n_elems * sizeof(float));
        memset(output_mems[i]->virt_addr, 0, output_attrs[i].n_elems * sizeof(float));
        // 4.2.2 Update input attrs
        output_attrs[i].type = RKNN_TENSOR_FLOAT32;
        // 4.1.3 Set output buffer
        rknn_set_io_mem(ctx, output_mems[i], &(output_attrs[i]));
    }
#else
    void *in_data = malloc(m_in_width * m_in_height * m_in_channel);
    memset(in_data, 0, m_in_width * m_in_height * m_in_channel);
    g_rga_dst = wrapbuffer_virtualaddr(in_data, m_in_width, m_in_height, RK_FORMAT_RGB_888);
#endif

    created = true;

    LOGI("rknn_init success!");

    return 0;
}

void destroy() {
//    LOGI("rknn_destroy!");
    // release io_mem resource
    for (int i = 0; i < n_input; ++i) {
        rknn_destroy_mem(ctx, input_mems[i]);
    }
    for (int i = 0; i < n_output; ++i) {
        rknn_destroy_mem(ctx, output_mems[i]);
    }
    rknn_destroy(ctx);
}

//bool run_yolo(char *inDataRaw, char *y0, char *y1, char *y2){
//    // Run
//    int ret;
//    printf("rknn_run\n");
//    ret = rknn_run(ctx, nullptr);
//    if (ret < 0) {
//        printf("rknn_run fail! ret=%d\n", ret);
//        return -1;
//    }
//
//    // Get Output
//    rknn_output outputs[1];
//    memset(outputs, 0, sizeof(outputs));
//    outputs[0].want_float = 1;
//    ret                   = rknn_outputs_get(ctx, 1, outputs, NULL);
//    if (ret < 0) {
//        printf("rknn_outputs_get fail! ret=%d\n", ret);
//        return -1;
//    }
//
//    // Post Process
//    for (int i = 0; i < io_num.n_output; i++) {
//        uint32_t MaxClass[5];
//        float    fMaxProb[5];
//        float*   buffer = (float*)outputs[i].buf;
//        uint32_t sz     = outputs[i].size / 4;
//
//        rknn_GetTop(buffer, fMaxProb, MaxClass, sz, 5);
//
//        printf(" --- Top5 ---\n");
//        for (int i = 0; i < 5; i++) {
//            printf("%3d: %8.6f\n", MaxClass[i], fMaxProb[i]);
//        }
//    }
//
//    // Release rknn_outputs
//    rknn_outputs_release(ctx, 1, outputs);
//
//    // Release
//    if (ctx > 0)
//    {
//        rknn_destroy(ctx);
//    }
//    if (model) {
//        free(model);
//    }
//    return true;
//}

cv::Mat bytesToMat(const unsigned char* data, int width, int height) {
    cv::Mat mat(height, width, CV_8UC3, const_cast<unsigned char*>(data));
    return mat;  // �����Ҫ�������ݶ����ǹ����ڴ棬��ʹ��clone()����
}
/*
 * ǰ����*/

return_d_m pre_process(cv::Mat& src, int input_size =192,bbox_landmark det= bbox_landmark()){
    cv::Mat dst;
    return_d_m dst_m3;
    int x1 = det.x1;
    int y1 = det.y1;
    int x2 = det.x2;
    int y2 = det.y2;
    int faceImgWidth = x2-x1;//��������
    int faceImgHeight = y2-y1;
    float center_w = x1+faceImgWidth/2; //ԭ�ȱ�ʾ������е�
    float center_h = y1+faceImgHeight/2;

//    double _scale = 0.28; //ԭscaleֵ�����������������Դ�С��ϵ����ֱ�Ӹ�ֵ0.3~0.5֮���ֵ
    double _scale = input_size / (MAX(faceImgWidth, faceImgHeight) * 1.5); //�����豸�ֱ��ʵ�������ϵ��

    cv::Mat m1 = cv::Mat::eye(cv::Size(3,2),CV_64F);   //��λ������scaling
    m1 *= _scale;
    // cout << m1 <<endl;

    cv::Mat m2 = cv::Mat::zeros(cv::Size(3,2),CV_64F);  //0������ƽ��
    m2.at<double >(0,2) = -(center_w * _scale) + input_size/2;
    m2.at<double >(1,2) = -(center_h * _scale) + input_size/2;
    // cout << m2 <<endl;

    dst_m3.matri = m1 + m2; // �������
    // cout << m3 <<endl;
    cv::warpAffine(src, dst_m3.dst,  dst_m3.matri, cv::Size(192,192));  //��� 192x192������Ϊģ������

    // imshow("img", dst_m3.dst);
    // imshow("origin img", src);
    // while(true)
    // {
    // int c = waitKey(20);
    // if(27==(char) c)
    // break;
    // }

    return dst_m3; //���ؽṹ�壬����ǰ������ͼƬ,�������
}

/*
����
*/
cv::Mat post_progress (cv::Mat& post_mat, cv::Mat& m){
    cv::Mat im;   //���������
    cv::Mat pts(106, 2, CV_32F);  //192x192�µ�����
    cv::Mat new_pts;
    cv::Mat im_t;
    cv::Mat im_t_f;
    cv::Mat coord;

    cv::Mat col_cat = cv::Mat::ones(106,1,CV_32F);

//    pts = post_mat.reshape(2, 106);
//    cv::resize(post_mat, pts, cv::Size(2,106));
    int index = 0;
    for (int i = 0; i < 106; ++i) {
        for (int j = 0; j < 2; ++j){//2��
//            LOGI("yyx pts[%d,%d]=%f",i,j,pts.at<float>(i,j));
            pts.at<float>(i,j) = post_mat.at<float>(index,0);
//            LOGI("yyx post_mat[%d,%0]=%f",index,j,post_mat.at<float>(index,0));
            index ++;
        }
    }


    invertAffineTransform(m, im);      //���������
    pts += 1;
    pts *= 192/2; //��ԭ��192x192�ߴ��µ�����ֲ�

//    LOGE("yyx pts�� x,y =%d,%d col_cat��x,y =%d,%d new_pts:x,y =%d,%d ", pts.rows,pts.cols,col_cat.rows,col_cat.cols,new_pts.rows,new_pts.cols);
//    LOGE("yyx pts�� dims =%d, col_cat��dims =%d ", pts.dims,col_cat.dims);
    cv::hconcat(pts, col_cat, new_pts);

    transpose(im, im_t); //ת�ò���
    im_t.convertTo(im_t_f,CV_32F); //ת�þ���תfloat����
    coord = new_pts * im_t_f;  //ת��Ϊ��������ֵ

    return coord;
}
// �� char* ����ת��Ϊ Mat ���󲢽���Ϊ float64 ��������
cv::Mat convertCharToFloat64(char* data, int rows, int cols) {
    // ����һ�� Mat ����ָ����������Ϊ float64
    cv::Mat mat(rows, cols, CV_64F);

    // �� char* ���ݸ��Ƶ� Mat ������
    memcpy(mat.data, data, sizeof(double) * rows * cols);

    return mat;
}

// �� float* ����ת��Ϊ Mat ���󲢽���Ϊ float64 ��������
cv::Mat convertFloatToFloat32(float* data, int rows, int cols) {
    // ����һ�� Mat ����ָ����������Ϊ float64
    cv::Mat mat(rows, cols, CV_32F);

    // �� float* ���ݸ��Ƶ� Mat ������
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat.at<float>(i, j) = static_cast<float>(data[i * cols + j]);
//            LOGE("mat.at<float>(%d, %d)=%f",i,j,mat.at<float>(i, j));
        }
    }

    return mat;
}

landmarks_output run_yolo(char *inDataRaw,float *y0,float *y1, char *y2,bbox_landmark det)
{
    float *ytemp = new float[212];
    landmarks_output result;
    int ret;
    float *p = new float [212];
    result.status = false;
    if(!created) {
        LOGE("run_yolo: init yolo hasn't successful!");
//        return false;
    }

#ifdef EVAL_TIME
    struct timeval start_time, stop_time;

    gettimeofday(&start_time, NULL);
#endif
    struct timeval start_time, stop_time;
    gettimeofday(&start_time, NULL); //模型前处理开始时间
    g_rga_src = wrapbuffer_virtualaddr((void *)inDataRaw, img_width, img_height,
                                       RK_FORMAT_RGBA_8888);

    cv::Mat origImage(img_height,img_width,CV_8UC3);
//    cv::Mat origImage(img_height,img_width,CV_8UC3, const_cast<char*>(inDataRaw));
    memcpy(origImage.data, inDataRaw,img_height*img_width*3);

    return_d_m image_str = pre_process(origImage,192,det);
    cv::Mat image  = image_str.dst;
    cv::Mat imageMatri = image_str.matri;
    uchar *pre_addr  =image.data;
    memcpy((void *)input_mems[0]->virt_addr, pre_addr, 192*192*3);
//    for (int i = 0; i < 150; i++) {
//        LOGI("yyx virt_addr[%d]=%d",i,((char *)input_mems[0]->virt_addr)[i]);
//    }

    // convert color format and resize. RGA8888 -> RGB888
//    ret = imresize(g_rga_src, g_rga_dst);
//    if (IM_STATUS_SUCCESS != ret) {
//        LOGE("run_yolo: resize image with rga failed: %s\n", imStrError((IM_STATUS)ret));
//        return false;
//    }
    gettimeofday(&stop_time, NULL);//前处理结束时间
    LOGI("face_landmark_preprocess use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);
#ifdef EVAL_TIME
    gettimeofday(&stop_time, NULL);
    LOGI("imresize use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);
#endif

#ifdef DEBUG_DUMP
    // save resized image
    if (g_inf_count == 5) {
        char out_img_name[1024];
        memset(out_img_name, 0, sizeof(out_img_name));
        sprintf(out_img_name, "/data/user/0/com.rockchip.gpadc.yolodemo/cache/resized_img_%d.rgb", g_inf_count);
        FILE *fp = fopen(out_img_name, "w");
//        LOGI("n_elems: %d", input_attrs[0].n_elems);
//        fwrite(input_mems[0]->virt_addr, 1, input_attrs[0].n_elems * sizeof(unsigned char), fp);
//        fflush(fp);
        for (int i = 0; i < input_attrs[0].n_elems; ++i) {
            fprintf(fp, "%d\n", *((uint8_t *)(g_rga_dst.vir_addr) + i));
        }
        fclose(fp);
    }

#endif

#if ZERO_COPY
#else
    rknn_input inputs[1];
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = m_in_width * m_in_height * m_in_channel;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    inputs[0].buf = g_rga_dst.vir_addr;
#ifdef EVAL_TIME
    gettimeofday(&start_time, NULL);
#endif
    rknn_inputs_set(ctx, 1, inputs);
#ifdef EVAL_TIME
    gettimeofday(&stop_time, NULL);
    LOGI("rknn_inputs_set use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);
#endif
#endif

#ifdef EVAL_TIME
    gettimeofday(&start_time, NULL);
#endif
    gettimeofday(&start_time, NULL);//推理开始时间
    ret = rknn_run(ctx, nullptr);
    if(ret < 0) {
        LOGE("rknn_run fail! ret=%d\n", ret);
//        return false;
    }
    gettimeofday(&stop_time, NULL); //推理结束时间
    LOGI("face_landmark_infer use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);
#ifdef EVAL_TIME
    gettimeofday(&stop_time, NULL);
    LOGI("inference use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);

    // outputs format are all NCHW.
    gettimeofday(&start_time, NULL);
#endif

#if ZERO_COPY
    gettimeofday(&start_time, NULL);
    float a = 3.14f;
//    LOGI("a=%f",a)
    memcpy(p, output_mems[0]->virt_addr, output_attrs[0].n_elems * sizeof(float));

    cv::Mat float32Mat = convertFloatToFloat32(p,212,1);
    // �� float* ���ݸ��Ƶ� Mat ������
//    for (int i = 0; i < 212; ++i) {
//        for (int j = 0; j < 1; ++j) {
//            LOGE("float32Mat(%d, %d)=%f",i,j,float32Mat.at<float>(i, j));
//        }
//    }
//    cv::Mat Image2 = cv::Mat(cv::Size(1,212),CV_64F);
//    memcpy(Image2.data, p, 212*4);
    cv::Mat post_pro = post_progress(float32Mat,imageMatri);
//    const float* data = post_pro.ptr<float>(0);
    float* data = post_pro.ptr<float>(0);

//    int x = 0;
//    for (int i = 0; i < 106; ++i) {
//        for (int j = 0; j < 2; ++j){//2��
//            LOGI("yyx pts[%d,%d]=%f",i,j,post_pro.at<float>(i,j));
//            LOGI("yyx data[%d]=%f",x,data[x]);
//            x++;
//        }
//    }
    memcpy(ytemp, data, 212 * sizeof(float));
//    memcpy(y1, output_mems[1]->virt_addr, output_attrs[1].n_elems * sizeof(char));
//    memcpy(y2, output_mems[2]->virt_addr, output_attrs[2].n_elems * sizeof(char));

//    LOGI("output_attrs[0].dims[0]: %d, output_attrs[0].dims[1]: %d, output_attrs[0].dims[2]: %d", output_attrs[0].dims[0], output_attrs[0].dims[1], output_attrs[0].dims[2])
//    LOGI("output_attrs[0].size: %d, output_attrs[0].type: %d, output_attrs[2].size: %d, ", output_attrs[0].size, output_attrs[0].type, output_attrs[2].size)
//
//    LOGI("output_attrs[0]_n_elems: %d", output_attrs[0].n_elems)
//    LOGI("output_attrs[1]_n_elems: %d", output_attrs[1].n_elems)
//    LOGI("output_attrs[2]_n_elems: %d", output_attrs[2].n_elems)
//    LOGI("sizeof(float): %d", sizeof(float))
    gettimeofday(&stop_time, NULL);
    LOGI("face_landmark_postprocess use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);
#else
    rknn_output outputs[3];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < 3; ++i) {
        outputs[i].want_float = 0;
    }
    rknn_outputs_get(ctx, 3, outputs, NULL);
    memcpy(y0, outputs[0].buf, output_attrs[0].n_elems * sizeof(char));
    memcpy(y1, outputs[1].buf, output_attrs[1].n_elems * sizeof(char));
    memcpy(y2, outputs[2].buf, output_attrs[2].n_elems * sizeof(char));
    rknn_outputs_release(ctx, 3, outputs);
#endif

#ifdef EVAL_TIME
    gettimeofday(&stop_time, NULL);
    LOGI("copy output use %f ms\n", (__get_us(stop_time) - __get_us(start_time)) / 1000);
#endif

#ifdef DEBUG_DUMP
    if (g_inf_count == 5) {
        for (int i = 0; i < n_output; ++i) {
            char out_path[1024];
            memset(out_path, 0, sizeof(out_path));
            sprintf(out_path, "/data/user/0/com.rockchip.gpadc.yolodemo/cache/out_%d.tensor", i);
            FILE *fp = fopen(out_path, "w");
            for (int j = 0; j < output_attrs[i].n_elems; ++j) {
#if ZERO_COPY
                fprintf(fp, "%d\n", *((int8_t *)(output_mems[i]->virt_addr) + i));
#else
                fprintf(fp, "%d\n", *((int8_t *)(outputs[i].buf) + i));
#endif
            }
            fclose(fp);
        }
    }
    if (g_inf_count < 10) {
        g_inf_count++;
    }
#endif

    result.status = true;
    result.face_landmark = ytemp;

    return result;
}



int yolo_post_process(char *grid0_buf, char *grid1_buf, char *grid2_buf,
                      int *ids, float *scores, float *boxes) {
    LOGI("*grid0_buf: %c, *grid1_buf: %c, *grid2_buf: %c", *grid0_buf, *grid1_buf, *grid2_buf);
    LOGI("*ids: %d,*scores: %f, *boxes: %f", *ids, *scores, *boxes);

    int ret;
    if(!created) {
        LOGE("yolo_post_process: init yolo hasn't successful!");
        return false;
    }

    detect_result_group_t detect_result_group;
//    LOGI("start yolo post.");
    ret = post_process((int8_t *)grid0_buf, (int8_t *)grid1_buf, (int8_t *)grid2_buf,
                       m_in_height, m_in_width, BOX_THRESH, NMS_THRESH, scale_w, scale_h,
                       out_zps, out_scales, &detect_result_group);
    if (ret < 0) {
        LOGE("yolo_post_process: post process failed!");
        return -1;
    }
//    LOGI("deteced %d objects.\n", detect_result_group.count);

    memset(ids, 0, sizeof(int) * OBJ_NUMB_MAX_SIZE);
    memset(scores, 0, sizeof(float) * OBJ_NUMB_MAX_SIZE);
    memset(boxes, 0, sizeof(float) * OBJ_NUMB_MAX_SIZE * BOX_LEN);

    int count = detect_result_group.count;
    for (int i = 0; i < count; ++i) {
        ids[i] = detect_result_group.results[i].class_id;
        scores[i] = detect_result_group.results[i].prop;
        *(boxes+4*i+0) = detect_result_group.results[i].box.left;
        *(boxes+4*i+1) = detect_result_group.results[i].box.top;
        *(boxes+4*i+2) = detect_result_group.results[i].box.right;
        *(boxes+4*i+3) = detect_result_group.results[i].box.bottom;
#ifdef DEBUG_DUMP
        if (g_post_count == 5) {
            LOGI("result %2d: (%4d, %4d, %4d, %4d), %d\n", i,
                 detect_result_group.results[i].box.left,
                 detect_result_group.results[i].box.top,
                 detect_result_group.results[i].box.right,
                 detect_result_group.results[i].box.bottom,
                 detect_result_group.results->class_id)
        }
        if (g_post_count < 10) {
            g_post_count++;
        }
#endif
    }

    return count;
}

int colorConvertAndFlip(void *src, int srcFmt, void *dst,  int dstFmt, int width, int height, int flip) {
    int ret;
    // RGA needs to ensure page alignment when using virtual addresses, otherwise it may cause
    // internal cache flushing errors. Manually modify src/dst buf to force its 4k alignment.
    // TODO -- convert color format and flip with OpenGL.
    int src_len = width * height * 3 / 2;    // yuv420 buffer length.
    void *src_ = malloc(src_len + 4096);
    void *org_src = src_;
    memset(src_, 0, src_len + 4096);
    src_ = (void *)((((int64_t)src_ >> 12) + 1) << 12);
    memcpy(src_, src, src_len);
    int dst_len = width * height * 4;    // rgba buffer length.
    void *dst_ = malloc(dst_len + 4096);
    void *org_dst = dst_;
    memset(dst_, 0, dst_len + 4096);
    dst_ = (void *)((((int64_t)dst_ >> 12) + 1) << 12);
    rga_buffer_t rga_src = wrapbuffer_virtualaddr((void *)src_, width, height, srcFmt);
    rga_buffer_t rga_dst = wrapbuffer_virtualaddr((void *)dst_, width, height, dstFmt);

    if (DO_NOT_FLIP == flip) {
        // convert color format
        ret = imcvtcolor(rga_src, rga_dst, rga_src.format, rga_dst.format);
    } else {
        // convert color format and flip.
        ret = imflip(rga_src, rga_dst, flip);
    }

    if (IM_STATUS_SUCCESS != ret) {
        LOGE("colorConvertAndFlip failed. Ret: %s\n", imStrError((IM_STATUS)ret));
    }

    memcpy(dst, dst_, dst_len);
    free(org_src);
    free(org_dst);

    return ret;
}

void rknn_app_destory() {
    LOGI("rknn app destroy.\n");
    if (g_rga_dst.vir_addr) {
        free(g_rga_dst.vir_addr);
    }
    rknn_destroy(ctx);
}
