/**
  * @ClassName yolo_image
  * @Description TODO
  * @Author raul.rao
  * @Date 2022/5/23 11:43
  * @Version 1.0
  */
#ifndef RK_YOLOV5_DEMO_YOLO_FACE_IMAGE_H
#define RK_YOLOV5_DEMO_YOLO_FACE_IMAGE_H

#include <android/log.h>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "rkyolo4j", ##__VA_ARGS__);
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "rkyolo4j", ##__VA_ARGS__);

int create_face(int im_height, int im_width, int im_channel, char *model_path);
void destroy_face();
bool run_yolo_face(char *inDataRaw, float *y0, float *y1, char *y2);

#endif //RK_YOLOV5_DEMO_YOLO_FACE_IMAGE_H
