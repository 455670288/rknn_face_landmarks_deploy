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

#include "yolo_face_image.h"
#include "yolo_image.h"
#include "rga/rga.h"
#include "rga/im2d.h"
#include "rga/im2d_version.h"
#include "post_process.h"


#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
//#include "opencv2/highgui.hpp"
//#include "opencv2/highgui_c.h"


//#define DEBUG_DUMP
//#define EVAL_TIME
#define ZERO_COPY 1
#define DO_NOT_FLIP -1

int g_inf_count_face = 0;

int g_post_count_face = 0;

rknn_context ctx_face = 0;

bool created_face = false;

int img_width_face = 0;    // the width of the actual input image
int img_height_face = 0;   // the height of the actual input image

int m_in_width_face = 0;   // the width of the RKNN model_face input
int m_in_height_face = 0;  // the height of the RKNN model_face input
int m_in_channel_face = 0; // the channel of the RKNN model_face input

float scale_w_face = 0.0;
float scale_h_face = 0.0;

uint32_t n_input_face = 1;
uint32_t n_output_face = 9;

rknn_tensor_attr input_attrs_face[1];
rknn_tensor_attr output_attrs_face[9]; //大小必须要跟模型输出个数相同

rknn_tensor_mem *input_mems_face[1];
rknn_tensor_mem *output_mems_face[9]; //大小必须要跟模型输出个数相同

rga_buffer_t g_rga_src_face;
rga_buffer_t g_rga_dst_face;
unsigned char *model_face;
rknn_input_output_num io_num_face;

std::vector<float> out_scales_face;
std::vector<int32_t> out_zps_face;

double __get_us_face(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }


int create_face(int im_height, int im_width, int im_channel, char *model_path)
{
    img_height_face = im_height;
    img_width_face = im_width;

    LOGI("try  face rknn_init!");

    // 0. RGA version check
    LOGI("face RGA API Version: %s", RGA_API_VERSION);
    // Please refer to the link to confirm the RGA driver version, make sure it is higher than 1.2.4
    // https://github.com/airockchip/librga/blob/main/docs/Rockchip_FAQ_RGA_CN.md#rga-driver

    // 1. Load model_face
    FILE *fp = fopen(model_path, "rb");
    if(fp == NULL) {
        LOGE("fopen %s fail!\n", model_path);
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    uint32_t model_len = ftell(fp);
    void *model_face = malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if(model_len != fread(model_face, 1, model_len, fp)) {
        LOGE("fread %s fail!\n", model_path);
        free(model_face);
        fclose(fp);
        return -1;
    }

    fclose(fp);

    // 2. Init RKNN model_face
    int ret = rknn_init(&ctx_face, model_face, model_len, 0, nullptr);
    free(model_face);

    if(ret < 0) {
        LOGE("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // 3. Query input/output attr.
    rknn_input_output_num io_num_face;
    rknn_query_cmd cmd = RKNN_QUERY_IN_OUT_NUM;
    // 3.1 Query input/output num.
    ret = rknn_query(ctx_face, cmd, &io_num_face, sizeof(io_num_face));
    if (ret != RKNN_SUCC) {
        LOGE("rknn_query io_num_face fail!ret=%d\n", ret);
        return -1;
    }
    n_input_face = io_num_face.n_input;
    n_output_face = io_num_face.n_output;

    // 3.2 Query input attributes
    memset(input_attrs_face, 0, n_input_face * sizeof(rknn_tensor_attr));
    for (int i = 0; i < n_input_face; ++i) {
        input_attrs_face[i].index = i;
        cmd = RKNN_QUERY_INPUT_ATTR;
        ret = rknn_query(ctx_face, cmd, &(input_attrs_face[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            LOGE("rknn_query input_attrs_face[%d] fail!ret=%d\n", i, ret);
            return -1;
        }
    }
    // 3.2.0 Update global model_face input shape.
    if (RKNN_TENSOR_NHWC == input_attrs_face[0].fmt) {
        m_in_height_face = input_attrs_face[0].dims[1];
        m_in_width_face = input_attrs_face[0].dims[2];
        m_in_channel_face = input_attrs_face[0].dims[3];
    } else if (RKNN_TENSOR_NCHW == input_attrs_face[0].fmt) {
        m_in_height_face = input_attrs_face[0].dims[2];
        m_in_width_face = input_attrs_face[0].dims[3];
        m_in_channel_face = input_attrs_face[0].dims[1];
    } else {
        LOGE("Unsupported model_face input layout: %d!\n", input_attrs_face[0].fmt);
        return -1;
    }

    // set scale_w_face, scale_h_face for post process
    scale_w_face = (float)m_in_width_face / img_width_face;
    scale_h_face = (float)m_in_height_face / img_height_face;
    LOGI("m_in_width_face = %d", m_in_width_face)
    LOGI("img_width_face = %d", img_width_face)
    LOGI("scale_w_face = %f", scale_w_face)
    LOGI("m_in_height_face = %d", m_in_height_face)
    LOGI("img_height_face = %d", img_height_face)
    LOGI("scale_h_face = %f", scale_h_face)
    LOGI("n_output_face = %d", n_output_face)
    LOGI("sizeof(rknn_tensor_attr) = %d", sizeof(rknn_tensor_attr))

    // 3.3 Query output attributes
    memset(output_attrs_face, 0, n_output_face * sizeof(rknn_tensor_attr));
    for (int i = 0; i < n_output_face; ++i) {
        output_attrs_face[i].index = i;
        cmd = RKNN_QUERY_OUTPUT_ATTR;
        ret = rknn_query(ctx_face, cmd, &(output_attrs_face[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) {
            LOGE("rknn_query output_attrs_face[%d] fail!ret=%d\n", i, ret);
            return -1;
        }
        // set out_scales_face/out_zps_face for post_process
        out_scales_face.push_back(output_attrs_face[i].scale);
        out_zps_face.push_back(output_attrs_face[i].zp);
    }

#if ZERO_COPY
    // 4. Set input/output buffer
    // 4.1 Set inputs memory
    // 4.1.1 Create input tensor memory, input data type is INT8, yolo has only 1 input.
    input_mems_face[0] = rknn_create_mem(ctx_face, input_attrs_face[0].size_with_stride * sizeof(char));
    memset(input_mems_face[0]->virt_addr, 0, input_attrs_face[0].size_with_stride * sizeof(char));
    // 4.1.2 Update input attrs
    input_attrs_face[0].index = 0;
    input_attrs_face[0].type = RKNN_TENSOR_UINT8;
    input_attrs_face[0].size = m_in_height_face * m_in_width_face * m_in_channel_face * sizeof(char);
    input_attrs_face[0].fmt = RKNN_TENSOR_NHWC;
    LOGI("input_attrs_face[0].size = %d", input_attrs_face[0].size)
    LOGI("m_in_height_face = %d", m_in_height_face)
    LOGI("m_in_width_face = %d", m_in_width_face)
    LOGI("m_in_channel_face = %d", m_in_channel_face)
    // TODO -- The efficiency of pass through will be higher, we need adjust the layout of input to
    //         meet the use condition of pass through.
    input_attrs_face[0].pass_through = 0;
    // 4.1.3 Set input buffer
    rknn_set_io_mem(ctx_face, input_mems_face[0], &(input_attrs_face[0]));
    // 4.1.4 bind virtual address to rga virtual address
    g_rga_dst_face = wrapbuffer_virtualaddr((void *)input_mems_face[0]->virt_addr, m_in_width_face, m_in_height_face,
                                       RK_FORMAT_RGB_888);

    LOGI("output_attrs_face[0].type = %d", output_attrs_face[0].type)
    // 4.2 Set outputs memory
    for (int i = 0; i < n_output_face; ++i) {
        // 4.2.1 Create output tensor memory, output data type is int8, post_process need int8 data.
        output_mems_face[i] = rknn_create_mem(ctx_face, output_attrs_face[i].n_elems * sizeof(float));
        memset(output_mems_face[i]->virt_addr, 0, output_attrs_face[i].n_elems * sizeof(float));
        // 4.2.2 Update input attrs
        output_attrs_face[i].type = RKNN_TENSOR_FLOAT32;
        // 4.1.3 Set output buffer
        rknn_set_io_mem(ctx_face, output_mems_face[i], &(output_attrs_face[i]));
    }
#else
    void *in_data = malloc(m_in_width_face * m_in_height_face * m_in_channel_face);
    memset(in_data, 0, m_in_width_face * m_in_height_face * m_in_channel_face);
    g_rga_dst_face = wrapbuffer_virtualaddr(in_data, m_in_width_face, m_in_height_face, RK_FORMAT_RGB_888);
#endif

    created_face = true;

    LOGI("rknn_init success!");

    return 0;
}

void destroy_face() {
//    LOGI("rknn_destroy!");
    // release io_mem resource
    for (int i = 0; i < n_input_face; ++i) {
        rknn_destroy_mem(ctx_face, input_mems_face[i]);
    }
    for (int i = 0; i < n_output_face; ++i) {
        rknn_destroy_mem(ctx_face, output_mems_face[i]);
    }
    rknn_destroy(ctx_face);
}


// �� char* ����ת��Ϊ Mat ���󲢽���Ϊ float64 ��������
cv::Mat convertCharToFloat64_face(char* data, int rows, int cols) {
    // ����һ�� Mat ����ָ����������Ϊ float64
    cv::Mat mat(rows, cols, CV_64F);

    // �� char* ���ݸ��Ƶ� Mat ������
    memcpy(mat.data, data, sizeof(double) * rows * cols);

    return mat;
}

// �� float* ����ת��Ϊ Mat ���󲢽���Ϊ float64 ��������
cv::Mat convertFloatToFloat32_face(float* data, int rows, int cols) {
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


/*
�������ǰ���� img:ԭʼ�ߴ�ͼƬ,  input_size:ģ�ͳߴ�360*360
*/
face_pre face_dete_pre (cv::Mat& img, int input_size = 240){
    face_pre face_pre_res;
    float im_ratio = static_cast<float>(img.rows)/ img.cols;
    float model_ratio =   1.0;

    int new_width, new_height;

    if(im_ratio > model_ratio){//���ֳ����
        new_height = input_size;
        new_width = static_cast<int> (new_height /im_ratio);
    }else{
        new_width = input_size;
        new_height = static_cast<int> (new_width * im_ratio);
    }

    face_pre_res.det_scale = static_cast<float>(new_height) / img.rows;
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(new_width, new_height));

    cv::Mat det_img = cv::Mat::zeros(cv::Size(input_size,input_size), CV_8UC3); //����ģ������ߴ�
    cv::Rect roi(0, 0, new_width, new_height);
    resized_img.copyTo(det_img(roi)); //roi��������img���ǣ���������padding
    face_pre_res.det_img = det_img;


    return face_pre_res;
}

/*
decoder
*/
cv::Mat distance2bbox(cv::Mat& points, cv::Mat& distance){
    cv::Mat x1 = points.col(0) - distance.col(0);
    cv::Mat y1 = points.col(1) - distance.col(1);
    cv::Mat x2 = points.col(0) + distance.col(2);
    cv::Mat y2 = points.col(1) + distance.col(3);

    cv::Mat bboxes;
    //����ƴ�ӣ��õ�bounding box
    cv::hconcat(x1,y1,bboxes);
    cv::hconcat(bboxes, x2, bboxes);
    cv::hconcat(bboxes, y2, bboxes);

    return bboxes;
}


/*
 nms
*/
std::vector<int> nms (const std::vector<bbox_landmark>& dets, float nms_thresh = 0.4){
    std::vector<int> keep;
    float thresh = nms_thresh;

    std::vector<float> x1, y1, x2, y2, scores;
    for(const auto& det: dets){
        x1.push_back(det.x1);
        y1.push_back(det.y1);
        x2.push_back(det.x2);
        y2.push_back(det.y2);
        scores.push_back(det.score);
    }
    std::vector<float> areas;
    for(int i =0; i < dets.size(); i++){
        areas.push_back((x2[i] - x1[i] + 1) * (y2[i] - y1[i] + 1));
    }

    std::vector<int> order(dets.size());
//    std::iota(order.begin(), order.end(), 0);
    int value = 0;
    for(int &num : order){
        num = value++;
    }
    std::sort(order.begin(), order.end(), [&](int a, int b){return scores[a] > scores[b];});

    while(!order.empty()){
        int i = order[0];
        keep.push_back(i);

        std::vector<int> inds;
        for (int j = 1; j < order.size(); j++){   //计算交集坐标
            float xx1 = std::max(x1[i], x1[order[j]]);
            float yy1 = std::max(y1[i] , y1[order[j]]);
            float xx2 = std::min(x2[i], x2[order[j]]);
            float yy2 = std::min(y2[i] , y2[order[j]]);

            float w = std::max(0.0f, xx2 -xx1 +1);
            float h = std::max(0.0f, yy2 - yy1 +1);
            float inter = w * h; //计算交集面积
            float ovr = inter /(areas[i] + areas[order[j]] - inter); //交并比

            if(ovr <= thresh){
                inds.push_back(j);// 存放多目标索引
            }
        }
        std::vector<int> new_order;
        for(int j = 0; j< inds.size(); j++){
            new_order.push_back(order[inds[j]]);
        }
        order = new_order;
    }
    return keep;
}

/*
����������
*/
std::vector<bbox_landmark> face_dete_pos (float threshold = 0.5, std::vector<cv::Mat> net_outs = std::vector<cv::Mat>(), float det_scale =0.3){
    std::vector<float> scores_list;
//    std::vector<cv::Rect> bboxes_list;
    std::vector<bbox_landmark> bboxes_list;
    std::vector<int> feat_stride_fpn = {8, 16 ,32}; //�²�������
    int fmc = 3; //ģ������Ĳ�������������ڣ�ÿ������������������
    int num_anchors = 2;


    for(int idx =0; idx < feat_stride_fpn.size(); ++idx){
        cv::Mat scores = net_outs[idx];
        cv::Mat bbox_preds = net_outs[idx + fmc];
        bbox_preds = bbox_preds * feat_stride_fpn[idx];
        int height = std::ceil(static_cast<float>(120) / feat_stride_fpn[idx]); //�²������������ȡ��
        int width = std::ceil(static_cast<float>(120) /feat_stride_fpn[idx]);//floor
        int K = height * width;
//        LOGE("K = %d!",K);
        // std::tuple<int, int, int> key(height, width, feat_stride_fpn[idx]); // Ԫ��洢���ߣ����²�������

        //����ê������
        cv::Mat anchor_centers(height, width ,CV_32FC2);
        for (int y=0 ; y< height; ++y){
            for (int x =0; x < width; ++x){
                anchor_centers.at<cv::Point2f>(y,x) = cv::Point2f(static_cast<float>(x), static_cast<float>(y));
            }
        }
        anchor_centers *= feat_stride_fpn[idx];  //�˲�������
        anchor_centers = anchor_centers.reshape(1, K).clone(); // K��2��

        cv::Mat repeated_anchor;   //K*2��2��
        cv::repeat(anchor_centers, num_anchors, 1, repeated_anchor);
        for (int i = 0; i < anchor_centers.rows; ++i) {
            anchor_centers.row(i).copyTo(repeated_anchor.row(i * 2));  // ���Ƶ�ż����
            anchor_centers.row(i).copyTo(repeated_anchor.row(i * 2 + 1));  // ���Ƶ�������
        }

        //scores��ֵɸѡ
        std::vector<int> pos_inds; //����ֵ
        for(int i =0; i < scores.total(); ++i){
            if(scores.at<float>(i) >= threshold){
                pos_inds.push_back(i); //��ȷ��pos_indsΪ��ʱ���Ƿ񱨴�
            }
        }
//        LOGE("0000000000000000000000!");
        //ê���ģ��Ԥ����룬ת��Ϊbbox����
        cv::Mat bboxes = distance2bbox(repeated_anchor, bbox_preds);  // bboxes: K*2�У�4��

        std::vector<float> pos_scores;
        std::vector<bbox_landmark> pos_bboxes;
//        LOGE("11111111111111111111!");
        for (const auto& index : pos_inds){    //���һ���߶ȿ�����з�����ֵ��Χ��scores��bboxes
            pos_scores.push_back(scores.at<float>(index));
            float x1 = bboxes.at<float>(index, 0) / det_scale;
            float y1 = bboxes.at<float>(index, 1) / det_scale;
            float x2 = bboxes.at<float>(index, 2) / det_scale;
            float y2 = bboxes.at<float>(index ,3) / det_scale;
            float score = scores.at<float>(index);
            bbox_landmark rect(x1,y1,x2,y2,score);
//            LOGE("222222222222222!");
//            LOGE("x1 = %f ,y1 = %f ,x2 = %f ,y2 = %f, score= %f",x1,y1,x2,y2,score);
            pos_bboxes.push_back(rect);
        }
//        LOGE("33333333333333!");
        scores_list.insert(scores_list.end(), pos_scores.begin(), pos_scores.end()); //�������scores��bboxes
        bboxes_list.insert(bboxes_list.end(), pos_bboxes.begin(), pos_bboxes.end());
//        LOGE("44444444444444444!");
    }

    std::vector<int> keep = nms(bboxes_list, 0.4); //keep存储模型检测到的多人bbox索引

//    for(int i = 0; i< keep.size(); i++){
//        LOGE("face[%d] \n",i);
//    }

    if(scores_list.empty()){
        LOGE("scores_list  is empty!");
    }

    /*
     存储nms后的所有检测框
     */
    std::vector<bbox_landmark> boxes;
    for(int nums : keep){
        boxes.push_back(bboxes_list[nums]);
    }


    /*��ʹ��nms,ֱ��ȡscore��ߵļ���*/
//    auto max_element_iterator = std::max_element(scores_list.begin(), scores_list.end());
//    int max_index = std::distance(scores_list.begin(), max_element_iterator);
//
//    bbox_landmark det ;
//    if(max_index > 0){
//        det = bboxes_list[max_index];
//    }
    return boxes;   //��cv::Rect���洢bbox��ֵ�� det.x : x1,  det.y : y1,  det.width : x2, det.height : y2
}

bool run_yolo_face(char *inDataRaw, float *y0, float *y1, char *y2)
{
//    run_yolo(inDataRaw,y0,y1,y2);
//    return true;

    int ret;
    bool status = false;
//    LOGE("run_yolo: init yolo  successful!");
    if(!created_face) {
        LOGE("run_yolo: init yolo hasn't successful!");
        return false;
    }

#ifdef EVAL_TIME
    struct timeval start_time, stop_time;

    gettimeofday(&start_time, NULL);
#endif
    struct timeval start_time, stop_time; // 初始化时间戳
    gettimeofday(&start_time, NULL); //推理开始时间
    g_rga_src_face = wrapbuffer_virtualaddr((void *)inDataRaw, img_width_face, img_height_face,
                                       RK_FORMAT_RGBA_8888);

    cv::Mat origImage(img_height_face,img_width_face,CV_8UC3);
//    cv::Mat origImage(img_height_face,img_width_face,CV_8UC3, const_cast<char*>(inDataRaw));
    memcpy(origImage.data, inDataRaw,img_height_face*img_width_face*3);
    face_pre image_str = face_dete_pre(origImage,120);
    cv::Mat image  = image_str.det_img;
    float imageMatri = image_str.det_scale;
    uchar *pre_addr  =image.data;
    memcpy((void *)input_mems_face[0]->virt_addr, pre_addr, 120*120*3);
    gettimeofday(&stop_time, NULL); //前处理结束时间
    LOGI("face_dete_pre_process use %f ms\n", (__get_us_face(stop_time) - __get_us_face(start_time)) / 1000);


#ifdef EVAL_TIME
    gettimeofday(&stop_time, NULL);
    LOGI("imresize use %f ms\n", (__get_us_face(stop_time) - __get_us_face(start_time)) / 1000);
#endif

#ifdef DEBUG_DUMP
    // save resized image
    if (g_inf_count_face == 5) {
        char out_img_name[1024];
        memset(out_img_name, 0, sizeof(out_img_name));
        sprintf(out_img_name, "/data/user/0/com.rockchip.gpadc.yolodemo/cache/resized_img_%d.rgb", g_inf_count_face);
        FILE *fp = fopen(out_img_name, "w");
//        LOGI("n_elems: %d", input_attrs_face[0].n_elems);
//        fwrite(input_mems_face[0]->virt_addr, 1, input_attrs_face[0].n_elems * sizeof(unsigned char), fp);
//        fflush(fp);
        for (int i = 0; i < input_attrs_face[0].n_elems; ++i) {
            fprintf(fp, "%d\n", *((uint8_t *)(g_rga_dst_face.vir_addr) + i));
        }
        fclose(fp);
    }

#endif

#if ZERO_COPY
#else
    rknn_input inputs[1];
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = m_in_width_face * m_in_height_face * m_in_channel_face;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    inputs[0].buf = g_rga_dst_face.vir_addr;
#ifdef EVAL_TIME
    gettimeofday(&start_time, NULL);
#endif
    rknn_inputs_set(ctx_face, 1, inputs);
#ifdef EVAL_TIME
    gettimeofday(&stop_time, NULL);
    LOGI("rknn_inputs_set use %f ms\n", (__get_us_face(stop_time) - __get_us_face(start_time)) / 1000);
#endif
#endif

#ifdef EVAL_TIME
    gettimeofday(&start_time, NULL);
#endif
    gettimeofday(&start_time, NULL); //推理开始时间
    ret = rknn_run(ctx_face, nullptr);
    if(ret < 0) {
        LOGE("rknn_run fail! ret=%d\n", ret);
        return false;
    }
    gettimeofday(&stop_time, NULL); //推理结束时间
    LOGI("face_dete_inference use %f ms\n", (__get_us_face(stop_time) - __get_us_face(start_time)) / 1000);
#ifdef EVAL_TIME
    gettimeofday(&stop_time, NULL);
    LOGI("inference use %f ms\n", (__get_us_face(stop_time) - __get_us_face(start_time)) / 1000);

    // outputs format are all NCHW.
    gettimeofday(&start_time, NULL);
#endif

#if ZERO_COPY
//    p =(float *) malloc(output_attrs_face[0].n_elems * sizeof(float));
//    memcpy(p, output_mems_face[0]->virt_addr, output_attrs_face[0].n_elems * sizeof(float));
//
//    LOGI("output_attrs_fac[0]_n_elems: %d", output_attrs_face[0].n_elems)
//    LOGI("output_attrs_face[1]_n_elems: %d", output_attrs_face[1].n_elems)
//    LOGI("output_attrs_face[2]_n_elems: %d", output_attrs_face[2].n_elems)
//    LOGI("output_attrs_face[3]_n_elems: %d", output_attrs_face[3].n_elems)
//    LOGI("output_attrs_face[4]_n_elems: %d", output_attrs_face[4].n_elems)
//    LOGI("output_attrs_face[5]_n_elems: %d", output_attrs_face[5].n_elems)
//    LOGI("output_attrs_face[6]_n_elems: %d", output_attrs_face[6].n_elems)
//    LOGI("output_attrs_face[7]_n_elems: %d", output_attrs_face[7].n_elems)
//    LOGI("output_attrs_face[8]_n_elems: %d", output_attrs_face[8].n_elems)
//    LOGI("output_attrs_face[9]_n_elems: %d", output_attrs_face[9].n_elems)
//    LOGI("output_attrs_face[10]_n_elems: %d", output_attrs_face[10].n_elems)
//    LOGI("output_attrs_face[11]_n_elems: %d", output_attrs_face[11].n_elems)
//    LOGI("output_attrs_face[12]_n_elems: %d", output_attrs_face[12].n_elems)
//    LOGI("output_attrs_face[13]_n_elems: %d", output_attrs_face[13].n_elems)
//    LOGI("output_attrs_face[14]_n_elems: %d", output_attrs_face[14].n_elems)


    gettimeofday(&start_time, NULL);
    std::vector<cv::Mat> imageVector;
    cv::Mat mat1 = convertFloatToFloat32_face((float *)output_mems_face[0]->virt_addr,output_attrs_face[0].n_elems,1);
    cv::Mat mat2 = convertFloatToFloat32_face((float *)output_mems_face[1]->virt_addr,output_attrs_face[1].n_elems,1);
    cv::Mat mat3 = convertFloatToFloat32_face((float *)output_mems_face[2]->virt_addr,output_attrs_face[2].n_elems,1);
    cv::Mat mat4 = convertFloatToFloat32_face((float *)output_mems_face[3]->virt_addr,output_attrs_face[3].n_elems/4,4);
    cv::Mat mat5 = convertFloatToFloat32_face((float *)output_mems_face[4]->virt_addr,output_attrs_face[4].n_elems/4,4);
    cv::Mat mat6 = convertFloatToFloat32_face((float *)output_mems_face[5]->virt_addr,output_attrs_face[5].n_elems/4,4);
    cv::Mat mat7 = convertFloatToFloat32_face((float *)output_mems_face[6]->virt_addr,output_attrs_face[6].n_elems/10,10);
    cv::Mat mat8 = convertFloatToFloat32_face((float *)output_mems_face[7]->virt_addr,output_attrs_face[7].n_elems/10,10);
    cv::Mat mat9 = convertFloatToFloat32_face((float *)output_mems_face[8]->virt_addr,output_attrs_face[8].n_elems/10,10);

//    cv::Mat mat1 = convertFloatToFloat32_face((float *)output_mems_face[0]->virt_addr,output_attrs_face[0].n_elems,1);
//    cv::Mat mat2 = convertFloatToFloat32_face((float *)output_mems_face[1]->virt_addr,output_attrs_face[1].n_elems,1);
//    cv::Mat mat3 = convertFloatToFloat32_face((float *)output_mems_face[2]->virt_addr,output_attrs_face[2].n_elems,1);
//    cv::Mat mat4 = convertFloatToFloat32_face((float *)output_mems_face[3]->virt_addr,output_attrs_face[3].n_elems,1);
//    cv::Mat mat5 = convertFloatToFloat32_face((float *)output_mems_face[4]->virt_addr,output_attrs_face[4].n_elems,1);
//
//    cv::Mat mat6 = convertFloatToFloat32_face((float *)output_mems_face[5]->virt_addr,output_attrs_face[5].n_elems/4,4);
//    cv::Mat mat7 = convertFloatToFloat32_face((float *)output_mems_face[6]->virt_addr,output_attrs_face[6].n_elems/4,4);
//    cv::Mat mat8 = convertFloatToFloat32_face((float *)output_mems_face[7]->virt_addr,output_attrs_face[7].n_elems/4,4);
//    cv::Mat mat9 = convertFloatToFloat32_face((float *)output_mems_face[8]->virt_addr,output_attrs_face[8].n_elems/4,4);
//    cv::Mat mat10 = convertFloatToFloat32_face((float *)output_mems_face[9]->virt_addr,output_attrs_face[9].n_elems/4,4);
//
//    cv::Mat mat11 = convertFloatToFloat32_face((float *)output_mems_face[10]->virt_addr,output_attrs_face[10].n_elems/10,10);
//    cv::Mat mat12 = convertFloatToFloat32_face((float *)output_mems_face[11]->virt_addr,output_attrs_face[11].n_elems/10,10);
//    cv::Mat mat13 = convertFloatToFloat32_face((float *)output_mems_face[12]->virt_addr,output_attrs_face[12].n_elems/10,10);
//    cv::Mat mat14 = convertFloatToFloat32_face((float *)output_mems_face[13]->virt_addr,output_attrs_face[13].n_elems/10,10);
//    cv::Mat mat15 = convertFloatToFloat32_face((float *)output_mems_face[14]->virt_addr,output_attrs_face[14].n_elems/10,10);


    imageVector.push_back(mat1);
    imageVector.push_back(mat2);
    imageVector.push_back(mat3);
    imageVector.push_back(mat4);
    imageVector.push_back(mat5);
    imageVector.push_back(mat6);
    imageVector.push_back(mat7);
    imageVector.push_back(mat8);
    imageVector.push_back(mat9);



    std::vector<bbox_landmark> det = face_dete_pos(0.2,imageVector,imageMatri);
//    for(int i =0; i < det.size(); i++){
//        LOGI("face[%d]_x1=[%f]", i,det[i].x1);//多人检测
//    }

    gettimeofday(&stop_time, NULL);
    LOGI("face_dete_post_process use %f ms\n", (__get_us_face(stop_time) - __get_us_face(start_time)) / 1000);
    std::vector<float*> landmarks_results;

    for(int i = 0; i < det.size(); i++){
        landmarks_output result = run_yolo(inDataRaw,y0,y1,y2,det[i]);
        landmarks_results.push_back(result.face_landmark);
    }

    for (int i = 0; i < landmarks_results.size(); i++) {
        if (i == 0){
            memcpy(y0, landmarks_results[i], 212 * sizeof(float));
        }
        else if (i == 1){
            memcpy(y1, landmarks_results[i], 212 * sizeof(float));
        }
//        memcpy(y0 + i * 212, landmarks_results[i], 212 * sizeof(float)); // 多人检测下的关键点数据拷贝到y0
        delete[] landmarks_results[i];
    }
    landmarks_results.clear();
    if (det.size() < 1){
        std::fill(y0, y0 + 212, -1); //屏幕没人时候全部置-1
        std::fill(y1, y1 + 212, -1);
    }
    LOGI("y0[10]=[%f],y1[10]=[%f]", y0[10], y1[10]);





//    cv::Mat float32Mat = convertFloatToFloat32_face(p,212,1);
//
//    cv::Mat post_pro = post_progress(float32Mat,imageMatri);
//    const float* data = post_pro.ptr<float>(0);
//
//    int x = 0;
//    for (int i = 0; i < 106; ++i) {
//        for (int j = 0; j < 2; ++j){//2��
//            LOGI("yyx pts[%d,%d]=%f",i,j,post_pro.at<float>(i,j));
//            LOGI("yyx data[%d]=%f",x,data[x]);
//            x++;
//        }
//    }
//    memcpy(y0, data, 212 * sizeof(float));
//    for (int i = 0; i < 212; ++i) {
//            LOGI("yyx y0[%d]=%f",i,y0[i]);
//    }
//    memcpy(y1, output_mems_face[1]->virt_addr, output_attrs_face[1].n_elems * sizeof(char));
//    memcpy(y2, output_mems_face[2]->virt_addr, output_attrs_face[2].n_elems * sizeof(char));
//
//    LOGI("output_attrs_face[0].dims[0]: %d, output_attrs_face[0].dims[1]: %d, output_attrs_face[0].dims[2]: %d", output_attrs_face[0].dims[0], output_attrs_face[0].dims[1], output_attrs_face[0].dims[2])
//    LOGI("output_attrs_face[0].size: %d, output_attrs_face[0].type: %d, output_attrs_face[2].size: %d, ", output_attrs_face[0].size, output_attrs_face[0].type, output_attrs_face[2].size)
//
//    LOGI("output_attrs_face[0]_n_elems: %d", output_attrs_face[0].n_elems)
//    LOGI("output_attrs_face[1]_n_elems: %d", output_attrs_face[1].n_elems)
//    LOGI("output_attrs_face[2]_n_elems: %d", output_attrs_face[2].n_elems)
//    LOGI("output_attrs_face[3]_n_elems: %d", output_attrs_face[3].n_elems)
//    LOGI("output_attrs_face[4]_n_elems: %d", output_attrs_face[4].n_elems)
//    LOGI("output_attrs_face[5]_n_elems: %d", output_attrs_face[5].n_elems)
//    LOGI("output_attrs_face[6]_n_elems: %d", output_attrs_face[6].n_elems)
//    LOGI("sizeof(float): %d", sizeof(float))
#else
    rknn_output outputs[3];
    memset(outputs, 0, sizeof(outputs));
    for (int i = 0; i < 3; ++i) {
        outputs[i].want_float = 0;
    }
    rknn_outputs_get(ctx_face, 3, outputs, NULL);
    memcpy(y0, outputs[0].buf, output_attrs_face[0].n_elems * sizeof(char));
    memcpy(y1, outputs[1].buf, output_attrs_face[1].n_elems * sizeof(char));
    memcpy(y2, outputs[2].buf, output_attrs_face[2].n_elems * sizeof(char));
    rknn_outputs_release(ctx_face, 3, outputs);
#endif

#ifdef EVAL_TIME
    gettimeofday(&stop_time, NULL);
    LOGI("copy output use %f ms\n", (__get_us_face(stop_time) - __get_us_face(start_time)) / 1000);
#endif

#ifdef DEBUG_DUMP
    if (g_inf_count_face == 5) {
        for (int i = 0; i < n_output_face; ++i) {
            char out_path[1024];
            memset(out_path, 0, sizeof(out_path));
            sprintf(out_path, "/data/user/0/com.rockchip.gpadc.yolodemo/cache/out_%d.tensor", i);
            FILE *fp = fopen(out_path, "w");
            for (int j = 0; j < output_attrs_face[i].n_elems; ++j) {
#if ZERO_COPY
                fprintf(fp, "%d\n", *((int8_t *)(output_mems_face[i]->virt_addr) + i));
#else
                fprintf(fp, "%d\n", *((int8_t *)(outputs[i].buf) + i));
#endif
            }
            fclose(fp);
        }
    }
    if (g_inf_count_face < 10) {
        g_inf_count_face++;
    }
#endif

    status = true;

//    LOGI("run_yolo: end\n");

    return status;
}


void rknn_app_destory_face() {
    LOGI("rknn app destroy.\n");
    if (g_rga_dst_face.vir_addr) {
        free(g_rga_dst_face.vir_addr);
    }
    rknn_destroy(ctx_face);
}
