package com.rockchip.gpadc.demo;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.ContentValues;
import android.content.Context;
import android.content.SharedPreferences;
import android.content.res.ColorStateList;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.hardware.Camera;
import android.opengl.GLES20;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.provider.MediaStore;
import android.support.v4.content.ContextCompat;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.SeekBar;
import android.widget.Toast;

import com.aglframework.smzh.AGLView;
import com.aglframework.smzh.CombineFilter;
import com.aglframework.smzh.IFilter;
import com.aglframework.smzh.camera.AGLCamera;
import com.aglframework.smzh.filter.FaceLiftFilter;
import com.aglframework.smzh.filter.GlassesFilterBackup;
import com.aglframework.smzh.filter.PupilFilterBackup;
import com.aglframework.smzh.filter.SimpleMakeupFilter;
import com.aglframework.smzh.filter.FaceDetector;
import com.aglframework.smzh.filter.LookupFilter;
import com.aglframework.smzh.filter.SimpleLipFilter;
import com.aglframework.smzh.filter.SmoothFilter;
import com.aglframework.smzh.filter.WhiteFilter;
import com.rockchip.gpadc.demo.yolo.InferenceWrapper;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static com.aglframework.smzh.CombineFilter.getCombineFilter;
import static com.rockchip.gpadc.demo.rga.HALDefine.CAMERA_PREVIEW_HEIGHT;
import static com.rockchip.gpadc.demo.rga.HALDefine.CAMERA_PREVIEW_WIDTH;
import static com.rockchip.gpadc.demo.yolo.PostProcess.INPUT_CHANNEL;

import javax.microedition.khronos.opengles.GL10;

public class MainActivityAGLcamera extends Activity implements Camera.PreviewCallback {
    private static final String TAG = MainActivityAGLcamera.class.getSimpleName();

    private AGLView aglView;
    private AGLCamera aglcamera;

    /** 滤镜组 用于合成所有妆效 */
    private CombineFilter combineFilter;
//    private SimpleLipFilter lipFilter;
//    private EyeFilter eyeFilter;
//    private SimpleMakeupFilter eyeShadowFiler;
    /** 美妆 */
    private SimpleMakeupFilter eyebrowFilter, lipFilter, blusherFilter, eyeShadowFiler, furrowFilter;
    /** 美瞳 */
    private PupilFilterBackup pupilFilter;
    private GlassesFilterBackup glassesFilter;
    /** 瘦脸 */
    private FaceLiftFilter faceLiftFilter;
    /** 美白 */
    private WhiteFilter whiteFilter;
    /** 磨皮 */
    private SmoothFilter smoothFilter;
    /** 滤镜 */
    private LookupFilter lutFilter;


    // for inference
    private String mModelName = "yolov5s.rknn";
    private String mModelNameFace = "yolov5s_face.rknn";
    private String platform = "rk3588";
    private InferenceWrapper mInferenceWrapper;
    private String fileDirPath;     // file dir to store model cache
    private ImageBufferQueue mImageBufferQueue;    // intermedia between camera thread and  inference thread
    private InferenceResult mInferenceResult = new InferenceResult();  // detection result
    private int mWidth;    //surface width
    private int mHeight;    //surface height
    private volatile boolean mStopInference = false;

    private ImageView mTrackResultView;
    private Bitmap mTrackResultBitmap = null;
    private Point[] points;

    private ImageView mButton;
    private ImageView show;
    private IFilter.Frame temp;
    private int t = 1;
    private boolean isPhotoSaved = false;
    private boolean showBeautyFunc = false;

    /** 美妆子项 */
    private ImageView mLighten, mSmooth, mThinner, mBlusher, mEyebrow, mLip, mPupil, mEyeShadow, mFurrow, mGlasses;
    /** 美妆功能键 */
    private ImageView mBeauty, mCompare;
    /** 美妆功能栏 按钮容器 */
    private LinearLayout mBeautyFunc;
    /** 美妆浓度 */
    private SeekBar mBeautyDensity;
    private List<ImageView> mBeautyFuncList;
    /** 美妆数据存储 */
    private SharedPreferences beautyData;
    /** 美妆数据 SharedPreferences 编辑器 */
    private SharedPreferences.Editor editor;

    private int count = 0;

    private Handler handler;
    private Object lock;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main_aglcamera);
        aglView = findViewById(R.id.camera_preview);
        mTrackResultView = (ImageView) findViewById(R.id.canvasView);

        mButton = (ImageView) findViewById(R.id.capture);
        show = (ImageView) findViewById(R.id.show);
        mButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                capture();

                // FIXME UI CHANGE2 新增关闭逻辑
                handler.postDelayed(new Runnable() {
                    @Override
                    public void run() {
                        show.setVisibility(View.GONE);
                    }
                }, 0);
            }
        });

        if (OpenCVLoader.initDebug()) {
            Log.d("yyx OpenCV", "OpenCV successfully loaded");
        } else {
            Log.d("yyx OpenCV", "OpenCV not loaded");
        }
        fileDirPath = getCacheDir().getAbsolutePath();

        platform = getPlatform();
        if (platform.equals("rk3588")) {
            createFile(mModelName, R.raw.change_rk3588_unquantized);
            createFile(mModelNameFace, R.raw.det_10g_change_rk3588_unquantized);

        } else if (platform.equals("rk356x")) {
            createFile(mModelName, R.raw.change_rk3566_unquantized);
            createFile(mModelNameFace, R.raw.det_500m_120_rk3566_unquan);
        }


        try {
            mInferenceResult.init(getAssets());
        } catch (IOException e) {
            e.printStackTrace();
        }
        mInferenceWrapper = new InferenceWrapper();

        handler = new Handler(getMainLooper());

        // FIXME UI CHANGE4 新增美妆数据
        beautyData = getSharedPreferences("beauty_data", Context.MODE_PRIVATE);
        editor = beautyData.edit();

        count = 0;
    }

    @Override
    protected void onResume() {
        super.onResume();

        // FIXME UI CHANGE5 新增美妆UI
        initBeautyUi();

        startTrack();

        if (aglcamera == null) {
//            1080不适用与平板
//            aglcamera = new AGLCamera(aglView,720,1280);
            aglcamera = new AGLCamera(aglView, 800, 600);
        }
        aglcamera.open();
        aglcamera.setCallback(this);

        lock = new Object();
        aglcamera.setLock(lock);

        t = 1;

        if (blusherFilter == null) {
            blusherFilter = new SimpleMakeupFilter(getApplicationContext());

            // 初始化相关数据
            blusherFilter.setStickerCoordinate(MakeupData.COORD_BLUSHER);
            blusherFilter.setIndices(MakeupData.INDICES_BLUSHER);
            blusherFilter.setCoordinateCount(44);
            blusherFilter.setTriangleCount(48);

            blusherFilter.setBitmap(BitmapFactory.decodeResource(getApplication().getResources(), R.drawable.blusher000));
            blusherFilter.setFaceDetector(new FaceDetector() {
                @Override
                public float[] onFaceDetected() {

                    float[] result = new float[88];

                    // 此处将 模型输出点 匹配到 OpenGL顶点
                    if (points != null) {
                        result[0] = (float) points[9].x;
                        result[1] = (float) points[9].y;
                        result[2] = (float) points[10].x;
                        result[3] = (float) points[10].y;
                        result[4] = (float) points[11].x;
                        result[5] = (float) points[11].y;
                        result[6] = (float) points[12].x;
                        result[7] = (float) points[12].y;
                        result[8] = (float) points[13].x;
                        result[9] = (float) points[13].y;
                        result[10] = (float) points[14].x;
                        result[11] = (float) points[14].y;
                        result[12] = (float) points[15].x;
                        result[13] = (float) points[15].y;
                        result[14] = (float) points[16].x;
                        result[15] = (float) points[16].y;
                        result[16] = (float) points[2].x;
                        result[17] = (float) points[2].y;
                        result[18] = (float) points[3].x;
                        result[19] = (float) points[3].y;
                        result[20] = (float) points[52].x;
                        result[21] = (float) points[52].y;
                        result[22] = (float) points[77].x;
                        result[23] = (float) points[77].y;
                        result[24] = (float) points[76].x;
                        result[25] = (float) points[76].y;
                        result[26] = (float) points[86].x;
                        result[27] = (float) points[86].y;
                        result[28] = (float) points[74].x;
                        result[29] = (float) points[74].y;
                        result[30] = (float) points[73].x;
                        result[31] = (float) points[73].y;
                        result[32] = (float) points[72].x;
                        result[33] = (float) points[72].y;
                        result[34] = (float) points[75].x;
                        result[35] = (float) points[75].y;
                        result[36] = (float) points[39].x;
                        result[37] = (float) points[39].y;
                        result[38] = (float) points[37].x;
                        result[39] = (float) points[37].y;
                        result[40] = (float) points[33].x;
                        result[41] = (float) points[33].y;
                        result[42] = (float) points[36].x;
                        result[43] = (float) points[36].y;
                        result[44] = (float) points[35].x;
                        result[45] = (float) points[35].y;
                        result[46] = (float) points[25].x;
                        result[47] = (float) points[25].y;
                        result[48] = (float) points[26].x;
                        result[49] = (float) points[26].y;
                        result[50] = (float) points[27].x;
                        result[51] = (float) points[27].y;
                        result[52] = (float) points[28].x;
                        result[53] = (float) points[28].y;
                        result[54] = (float) points[29].x;
                        result[55] = (float) points[29].y;
                        result[56] = (float) points[30].x;
                        result[57] = (float) points[30].y;
                        result[58] = (float) points[31].x;
                        result[59] = (float) points[31].y;
                        result[60] = (float) points[32].x;
                        result[61] = (float) points[32].y;
                        result[62] = (float) points[18].x;
                        result[63] = (float) points[18].y;
                        result[64] = (float) points[19].x;
                        result[65] = (float) points[19].y;
                        result[66] = (float) points[61].x;
                        result[67] = (float) points[61].y;
                        result[68] = (float) points[83].x;
                        result[69] = (float) points[83].y;
                        result[70] = (float) points[82].x;
                        result[71] = (float) points[82].y;
                        result[72] = (float) points[81].x;
                        result[73] = (float) points[81].y;
                        result[74] = (float) points[89].x;
                        result[75] = (float) points[89].y;
                        result[76] = (float) points[90].x;
                        result[77] = (float) points[90].y;
                        result[78] = (float) points[87].x;
                        result[79] = (float) points[87].y;
                        result[80] = (float) points[91].x;
                        result[81] = (float) points[91].y;
                        result[82] = (float) points[93].x;
                        result[83] = (float) points[93].y;
                        result[84] = (float) points[106].x;
                        result[85] = (float) points[106].y;
                        result[86] = (float) points[107].x;
                        result[87] = (float) points[107].y;
                        return result;
                    }

                    return null;
                }
            });

            blusherFilter.setStrength((float) beautyData.getInt("Blusher", 0) / 10);
        }

        if (lipFilter == null) {
            lipFilter = new SimpleMakeupFilter(getApplicationContext());

            // 初始化相关数据
            lipFilter.setStickerCoordinate(MakeupData.COORD_LIP);
            lipFilter.setIndices(MakeupData.INDICES_LIP);
            lipFilter.setCoordinateCount(36);
            lipFilter.setTriangleCount(54);

            lipFilter.setBitmap(BitmapFactory.decodeResource(getApplication().getResources(), R.drawable.lip2));
            lipFilter.setFaceDetector(new FaceDetector() {
                @Override
                public float[] onFaceDetected() {
                    // 数量对应纹理顶点数
                    float[] result = new float[72];

                    // 此处将 模型输出点 匹配到 OpenGL顶点
                    if (points != null) {
                        result[0] = (float) points[14].x;
                        result[1] = (float) points[14].y;   //	0
                        result[2] = (float) points[2].x;
                        result[3] = (float) points[2].y;    //	1
                        result[4] = (float) points[5].x;
                        result[5] = (float) points[5].y;    //	2
                        result[6] = (float) points[7].x;
                        result[7] = (float) points[7].y;    //	3
                        result[8] = (float) points[0].x;
                        result[9] = (float) points[0].y;    //	4
                        result[10] = (float) points[23].x;
                        result[11] = (float) points[23].y;    //	5
                        result[12] = (float) points[21].x;
                        result[13] = (float) points[21].y;    //	6
                        result[14] = (float) points[18].x;
                        result[15] = (float) points[18].y;    //	7
                        result[16] = (float) points[30].x;
                        result[17] = (float) points[30].y;    //	8
                        result[18] = (float) points[83].x;
                        result[19] = (float) points[83].y;    //	9
                        result[20] = (float) points[84].x;
                        result[21] = (float) points[84].y;    //	10
                        result[22] = (float) points[85].x;
                        result[23] = (float) points[85].y;    //	11
                        result[24] = (float) points[80].x;
                        result[25] = (float) points[80].y;    //	12
                        result[26] = (float) points[79].x;
                        result[27] = (float) points[79].y;    //	13
                        result[28] = (float) points[78].x;
                        result[29] = (float) points[78].y;    //	14
                        result[30] = (float) points[77].x;
                        result[31] = (float) points[77].y;    //	15
                        result[32] = (float) points[52].x;
                        result[33] = (float) points[52].y;    //	16
                        result[34] = (float) points[55].x;
                        result[35] = (float) points[55].y;    //	17
                        result[36] = (float) points[56].x;
                        result[37] = (float) points[56].y;    //	18
                        result[38] = (float) points[53].x;
                        result[39] = (float) points[53].y;    //	19
                        result[40] = (float) points[59].x;
                        result[41] = (float) points[59].y;    //	20
                        result[42] = (float) points[58].x;
                        result[43] = (float) points[58].y;    //	21
                        result[44] = (float) points[61].x;
                        result[45] = (float) points[61].y;    //	22
                        result[46] = (float) points[68].x;
                        result[47] = (float) points[68].y;    //	23
                        result[48] = (float) points[67].x;
                        result[49] = (float) points[67].y;    //	24
                        result[50] = (float) points[71].x;
                        result[51] = (float) points[71].y;    //	25
                        result[52] = (float) points[63].x;
                        result[53] = (float) points[63].y;    //	26
                        result[54] = (float) points[64].x;
                        result[55] = (float) points[64].y;    //	27
                        result[56] = (float) points[65].x;
                        result[57] = (float) points[65].y;    //	28
                        result[58] = (float) points[54].x;
                        result[59] = (float) points[54].y;    //	29
                        result[60] = (float) points[60].x;
                        result[61] = (float) points[60].y;    //	30
                        result[62] = (float) points[57].x;
                        result[63] = (float) points[57].y;    //	31
                        result[64] = (float) points[69].x;
                        result[65] = (float) points[69].y;    //	32
                        result[66] = (float) points[70].x;
                        result[67] = (float) points[70].y;    //	33
                        result[68] = (float) points[62].x;
                        result[69] = (float) points[62].y;    //	34
                        result[70] = (float) points[66].x;
                        result[71] = (float) points[66].y;    //	35
                        return result;
                    }

                    return null;
                }
            });

            lipFilter.setStrength((float) beautyData.getInt("Lip", 0) / 10);
        }

        if (eyebrowFilter == null) {
            eyebrowFilter = new SimpleMakeupFilter(getApplicationContext());

            // 初始化相关数据
            eyebrowFilter.setStickerCoordinate(MakeupData.COORD_EYE_BROW);
            eyebrowFilter.setIndices(MakeupData.INDICES_EYE_BROW);
            eyebrowFilter.setCoordinateCount(20);
            eyebrowFilter.setTriangleCount(18);

            eyebrowFilter.setBitmap(BitmapFactory.decodeResource(getApplication().getResources(), R.drawable.eye000));
            eyebrowFilter.setFaceDetector(new FaceDetector() {
                @Override
                public float[] onFaceDetected() {
                    // 数量对应纹理顶点数
                    float[] result = new float[40];
                    // 此处将 模型输出点 匹配到 OpenGL顶点
                    if (points != null) {
                        result[0] = (float) points[43].x;
                        result[1] = (float) points[43].y;    //	0
                        result[2] = (float) points[44].x;
                        result[3] = (float) points[44].y;    //	1
                        result[4] = (float) points[45].x;
                        result[5] = (float) points[45].y;    //	2
                        result[6] = (float) points[47].x;
                        result[7] = (float) points[47].y;    //	3
                        result[8] = (float) points[46].x;
                        result[9] = (float) points[46].y;    //	4
                        result[10] = (float) points[50].x;
                        result[11] = (float) points[50].y;    //	5
                        result[12] = (float) points[51].x;
                        result[13] = (float) points[51].y;    //	6
                        result[14] = (float) points[49].x;
                        result[15] = (float) points[49].y;    //	7
                        result[16] = (float) points[48].x;
                        result[17] = (float) points[48].y;    //	8
                        result[18] = (float) points[101].x;
                        result[19] = (float) points[101].y;    //	9
                        result[20] = (float) points[100].x;
                        result[21] = (float) points[100].y;    //	10
                        result[22] = (float) points[99].x;
                        result[23] = (float) points[99].y;    //	11
                        result[24] = (float) points[98].x;
                        result[25] = (float) points[98].y;    //	12
                        result[26] = (float) points[97].x;
                        result[27] = (float) points[97].y;    //	13
                        result[28] = (float) points[102].x;
                        result[29] = (float) points[102].y;    //	14
                        result[30] = (float) points[103].x;
                        result[31] = (float) points[103].y;    //	15
                        result[32] = (float) points[104].x;
                        result[33] = (float) points[104].y;    //	16
                        result[34] = (float) points[105].x;
                        result[35] = (float) points[105].y;    //	17
                        result[36] = (float) points[108].x;
                        result[37] = (float) points[108].y;    //	18
                        result[38] = (float) points[109].x;
                        result[39] = (float) points[109].y;    //	19
                        return result;
                    }

                    return null;
                }
            });

            eyebrowFilter.setStrength((float) beautyData.getInt("Eyebrow", 0) / 10);
        }

        if (eyeShadowFiler == null) {
            eyeShadowFiler = new SimpleMakeupFilter(getApplicationContext());

            // 初始化相关数据
            eyeShadowFiler.setStickerCoordinate(MakeupData.COORD_EYE_SHADOW);
            eyeShadowFiler.setIndices(MakeupData.INDICES_EYE_SHADOW);
            eyeShadowFiler.setCoordinateCount(36);
            eyeShadowFiler.setTriangleCount(41);

            eyeShadowFiler.setBitmap(BitmapFactory.decodeResource(getApplication().getResources(), R.drawable.eye1));
            eyeShadowFiler.setFaceDetector(new FaceDetector() {
                @Override
                public float[] onFaceDetected() {
                    // 数量对应纹理顶点数
                    float[] result = new float[72];
                    // 此处将 模型输出点 匹配到 OpenGL顶点
                    if (points != null) {
                        result[0] = (float) points[1].x;
                        result[1] = (float) points[1].y;
                        result[2] = (float) points[35].x;
                        result[3] = (float) points[35].y;
                        result[4] = (float) points[9].x;
                        result[5] = (float) points[9].y;
                        result[6] = (float) points[10].x;
                        result[7] = (float) points[10].y;
                        result[8] = (float) points[36].x;
                        result[9] = (float) points[36].y;
                        result[10] = (float) points[33].x;
                        result[11] = (float) points[33].y;
                        result[12] = (float) points[73].x;
                        result[13] = (float) points[73].y;
                        result[14] = (float) points[37].x;
                        result[15] = (float) points[37].y;
                        result[16] = (float) points[39].x;
                        result[17] = (float) points[39].y;
                        result[18] = (float) points[75].x;
                        result[19] = (float) points[75].y;
                        result[20] = (float) points[81].x;
                        result[21] = (float) points[81].y;
                        result[22] = (float) points[89].x;
                        result[23] = (float) points[89].y;
                        result[24] = (float) points[90].x;
                        result[25] = (float) points[90].y;
                        result[26] = (float) points[87].x;
                        result[27] = (float) points[87].y;
                        result[28] = (float) points[26].x;
                        result[29] = (float) points[26].y;
                        result[30] = (float) points[91].x;
                        result[31] = (float) points[91].y;
                        result[32] = (float) points[93].x;
                        result[33] = (float) points[93].y;
                        result[34] = (float) points[25].x;
                        result[35] = (float) points[25].y;
                        result[36] = (float) points[17].x;
                        result[37] = (float) points[17].y;
                        result[38] = (float) points[101].x;
                        result[39] = (float) points[101].y;
                        result[40] = (float) points[100].x;
                        result[41] = (float) points[100].y;
                        result[42] = (float) points[96].x;
                        result[43] = (float) points[96].y;
                        result[44] = (float) points[99].x;
                        result[45] = (float) points[99].y;
                        result[46] = (float) points[94].x;
                        result[47] = (float) points[94].y;
                        result[48] = (float) points[98].x;
                        result[49] = (float) points[98].y;
                        result[50] = (float) points[97].x;
                        result[51] = (float) points[97].y;
                        result[52] = (float) points[95].x;
                        result[53] = (float) points[95].y;
                        result[54] = (float) points[72].x;
                        result[55] = (float) points[72].y;
                        result[56] = (float) points[46].x;
                        result[57] = (float) points[46].y;
                        result[58] = (float) points[47].x;
                        result[59] = (float) points[47].y;
                        result[60] = (float) points[42].x;
                        result[61] = (float) points[42].y;
                        result[62] = (float) points[40].x;
                        result[63] = (float) points[40].y;
                        result[64] = (float) points[45].x;
                        result[65] = (float) points[45].y;
                        result[66] = (float) points[44].x;
                        result[67] = (float) points[44].y;
                        result[68] = (float) points[43].x;
                        result[69] = (float) points[43].y;
                        result[70] = (float) points[41].x;
                        result[71] = (float) points[41].y;
                        return result;
                    }

                    return null;
                }
            });

            eyeShadowFiler.setStrength((float) beautyData.getInt("Eyeshadow", 0) / 10);
        }

        if (pupilFilter == null) {
            pupilFilter = new PupilFilterBackup(getApplicationContext());
            pupilFilter.setBitmap(BitmapFactory.decodeResource(getApplication().getResources(), R.drawable.pupil1));
            pupilFilter.setFaceDetector(new PupilFilterBackup.FaceDetector() {
                @Override
                public float[] onFaceDetected(IFilter.Frame frame) {
                    float[] result = new float[16];
                    float widthL, widthR;
                    if (points != null && points.length > 0) {
                        widthL = ((float) points[37].x - (float) points[41].x) * 0.5f;
                        widthR = ((float) points[91].x - (float) points[95].x) * 0.5f-1f;
                        result[0] = (float) points[38].x - widthL;
                        result[1] = (float) points[38].y + widthL;
                        result[2] = (float) points[38].x + widthL;
                        result[3] = (float) points[38].y + widthL;
                        result[4] = (float) points[38].x - widthL;
                        result[5] = (float) points[38].y - widthL;
                        result[6] = (float) points[38].x + widthL;
                        result[7] = (float) points[38].y - widthL;

                        //R
                        result[8] = (float) points[88].x - widthR;
                        result[9] = (float) points[88].y + widthR;
                        result[10] = (float) points[88].x + widthR;
                        result[11] = (float) points[88].y + widthR;
                        result[12] = (float) points[88].x - widthR;
                        result[13] = (float) points[88].y - widthR;
                        result[14] = (float) points[88].x + widthR;
                        result[15] = (float) points[88].y - widthR;
                    }
                    return result;
                }
            });

            pupilFilter.setStrength((float) beautyData.getInt("Pupil", 0) / 10);
        }

        if (furrowFilter == null) {
            furrowFilter = new SimpleMakeupFilter(getApplicationContext());

            // 初始化相关数据
            furrowFilter.setStickerCoordinate(MakeupData.COORD_FURROW);
            furrowFilter.setIndices(MakeupData.INDICES_FURROW);
            furrowFilter.setCoordinateCount(15);
            furrowFilter.setTriangleCount(13);

            furrowFilter.setBitmap(BitmapFactory.decodeResource(getApplication().getResources(), R.drawable.wocan1));
            furrowFilter.setFaceDetector(new FaceDetector() {
                @Override
                public float[] onFaceDetected() {
                    // 数量对应纹理顶点数
                    float[] result = new float[30];
                    // 此处将 模型输出点 匹配到 OpenGL顶点
                    if (points != null) {
                        result[0] = (float)points[9].x;result[1] = (float)points[9].y;
                        result[2] = (float)points[10].x;result[3] = (float)points[10].y;
                        result[4] = (float)points[35].x;result[5] = (float)points[35].y;
                        result[6] = (float)points[36].x;result[7] = (float)points[36].y;
                        result[8] = (float)points[33].x;result[9] = (float)points[33].y;
                        result[10] = (float)points[37].x;result[11] = (float)points[37].y;
                        result[12] = (float)points[39].x;result[13] = (float)points[39].y;
                        result[14] = (float)points[73].x;result[15] = (float)points[73].y;
                        result[16] = (float)points[89].x;result[17] = (float)points[89].y;
                        result[18] = (float)points[90].x;result[19] = (float)points[90].y;
                        result[20] = (float)points[87].x;result[21] = (float)points[87].y;
                        result[22] = (float)points[91].x;result[23] = (float)points[91].y;
                        result[24] = (float)points[93].x;result[25] = (float)points[93].y;
                        result[26] = (float)points[25].x;result[27] = (float)points[25].y;
                        result[28] = (float)points[26].x;result[29] = (float)points[26].y;
                        return result;
                    }

                    return null;
                }
            });

            furrowFilter.setStrength((float) beautyData.getInt("Furrow", 0) / 10);
        }

        if (lutFilter == null) {
            lutFilter = new LookupFilter(getApplicationContext());
            lutFilter.setBitmap(BitmapFactory.decodeResource(getApplication().getResources(), R.drawable.lut_4));
            lutFilter.setIntensity(1.0f);
        }

        if (whiteFilter == null){
            whiteFilter = new WhiteFilter(getApplicationContext());
            whiteFilter.setWhiteLevel((float) beautyData.getInt("Lighten", 0) / 10);
        }

        if (smoothFilter == null){
            smoothFilter = new SmoothFilter(getApplicationContext());
            smoothFilter.setSmoothLevel((float) beautyData.getInt("Smooth", 0) / 10);
        }

        if (glassesFilter == null){
            glassesFilter = new GlassesFilterBackup(getApplicationContext());
            glassesFilter.setBitmap(BitmapFactory.decodeResource(getApplication().getResources(), R.drawable.glasses1));
            glassesFilter.setFaceDetector(new GlassesFilterBackup.FaceDetector() {
                @Override
                public float[] onFaceDetected(IFilter.Frame frame) {
                    float[] result = new float[8];
                    float width,height;
                    if(points!=null&&points.length>73){
                        width = ((float)points[25].x - (float) points[9].x)*0.5f+50f; height = width*0.5f;
                        result[0] = (float)points[73].x - width;result[1] = (float)points[73].y+height;
                        result[2] = (float)points[73].x + width;result[3] = (float)points[73].y+height;
                        result[4] = (float)points[73].x - width;result[5] = (float)points[73].y-height;
                        result[6] = (float)points[73].x + width;result[7] = (float)points[73].y-height;
                    }
                    return result;
                }
            });
        }

        if (faceLiftFilter == null){
            faceLiftFilter = new FaceLiftFilter(getApplicationContext());
            faceLiftFilter.setFaceDetector(new FaceDetector() {
                @Override
                public float[] onFaceDetected() {
                    float[] result = new float[236];

                    if (points != null) {
                        for (int i = 0; i < points.length; i++) {
                            int index = i * 2;
                            result[index] = (float) points[i].x;
                            result[index + 1] = (float) points[i].y;
                        }

                        return result;
                    }

                    return null;
                }
            });

            faceLiftFilter.setStrength((float) beautyData.getInt("Thinner", 0) / 10);
        }

        List<IFilter> filterList = new ArrayList<>();
        filterList.add(whiteFilter);
        filterList.add(smoothFilter);

        filterList.add(blusherFilter);
        filterList.add(lipFilter);
        filterList.add(eyebrowFilter);
        filterList.add(eyeShadowFiler);
        filterList.add(pupilFilter);
        filterList.add(furrowFilter);
        filterList.add(glassesFilter);

        filterList.add(faceLiftFilter);
        combineFilter = (CombineFilter) getCombineFilter(filterList);
        aglView.setFilter(combineFilter);

    }

    @Override
    protected void onPause() {
        super.onPause();
        if (aglcamera != null) {
            aglcamera.close();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();

        handler.removeCallbacksAndMessages(null) ;
        handler = null;
    }

    private void createFile(String fileName, int id) {
        String filePath = fileDirPath + "/" + fileName;
        try {
            File dir = new File(fileDirPath);

            if (!dir.exists()) {
                dir.mkdirs();
            }

            // 目录存在，则将apk中raw中的需要的文档复制到该目录下
            File file = new File(filePath);

            if (!file.exists()) {

                InputStream ins = getResources().openRawResource(id);// 通过raw得到数据资源
                FileOutputStream fos = new FileOutputStream(file);
                byte[] buffer = new byte[8192];
                int count = 0;

                while ((count = ins.read(buffer)) > 0) {
                    fos.write(buffer, 0, count);
                }

                fos.close();
                ins.close();

                Log.d("TAG", "Create " + filePath);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @SuppressLint("ClickableViewAccessibility")
    private void initBeautyUi(){
        mLighten = findViewById(R.id.iv_beauty_lighten);
        mLighten.setTag("Lighten");
        mSmooth = findViewById(R.id.iv_beauty_smooth);
        mSmooth.setTag("Smooth");
        mThinner = findViewById(R.id.iv_face_thinner);
        mThinner.setTag("Thinner");
        mBlusher = findViewById(R.id.iv_makeup_blusher);
        mBlusher.setTag("Blusher");
        mEyebrow = findViewById(R.id.iv_makeup_eyebrow);
        mEyebrow.setTag("Eyebrow");
        mLip = findViewById(R.id.iv_makeup_lips);
        mLip.setTag("Lip");
        mPupil = findViewById(R.id.iv_makeup_pupil);
        mPupil.setTag("Pupil");
        mEyeShadow = findViewById(R.id.iv_makeup_eyeshadow);
        mEyeShadow.setTag("Eyeshadow");
        mFurrow = findViewById(R.id.iv_makeup_furrow);
        mFurrow.setTag("Furrow");
        mGlasses = findViewById(R.id.iv_sticker_glasses);

        mBeauty = findViewById(R.id.btn_beauty);
        mCompare = findViewById(R.id.btn_compare);

        mBeautyFunc = findViewById(R.id.ll_makeup_list);
        mBeautyDensity = findViewById(R.id.sb_density);

        mBeautyFuncList = Arrays.asList(mLighten, mSmooth, mThinner, mBlusher, mEyebrow, mLip, mPupil, mEyeShadow, mFurrow);

        for (ImageView view : mBeautyFuncList){
            String tag = (String) view.getTag();
            int beautyDensity = beautyData.getInt(tag, -1);

            // 若未初始化
            if (beautyDensity == -1){
                editor.putInt(tag, 0);      // 初始化为0
                editor.apply();             // 并提交
                beautyDensity = 0;
            }

            // 若此前为设置过浓度，或者浓度为0，则置为未启用（灰色），否则选中（红色）
            view.setSelected(beautyDensity != 0);
        }

        //设置进度条颜色
        mBeautyDensity.setThumbTintList(ColorStateList.valueOf(ContextCompat.getColor(this, R.color.white)));
        mBeautyDensity.setProgressTintList(ColorStateList.valueOf(ContextCompat.getColor(this, R.color.white)));

        mBeauty.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (showBeautyFunc){
                    mBeautyFunc.setVisibility(View.GONE);
                    mBeautyDensity.setVisibility(View.GONE);

                    for (ImageView imageView : mBeautyFuncList){
                        // 处理功能状态
                        imageView.setActivated(false);
                        imageView.setSelected(beautyData.getInt((String) imageView.getTag(), 0) != 0);
                    }

                    mBeauty.setImageDrawable(getDrawable(R.drawable.icon_delight));
                    showBeautyFunc = false;
                } else {
                    mBeautyFunc.setVisibility(View.VISIBLE);
                    mBeauty.setImageDrawable(getDrawable(R.drawable.icon_close));
                    showBeautyFunc = true;
                }
            }
        });

        mCompare.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View view, MotionEvent motionEvent) {
                switch (motionEvent.getAction()){
                    case MotionEvent.ACTION_DOWN:
                        // 当手指按下屏幕时触发
                        view.setHovered(true);
                        aglView.setFilter(null);
                        return true; // 表示已经处理了这个事件

                    case MotionEvent.ACTION_UP:
                        // 当手指抬起屏幕时触发
                        view.setHovered(false);
                        aglView.setFilter(combineFilter);
                        return true; // 表示已经处理了这个事件
                }
                return false;
            }
        });

        // 定义子项监听器
        View.OnClickListener beautyFuncListener = new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                String tag = (String) view.getTag();
                int beautyDensity = beautyData.getInt(tag, -1);
//                Log.d(TAG, "点击了"+tag);

                for (ImageView imageView : mBeautyFuncList){
                    String ivTag = (String) imageView.getTag();
                    if (!tag.equals(ivTag)){
                        // 移除其余功能的启用状态（若持有浓度，则Selected仍为true，呈红色）
                        imageView.setActivated(false);

                        // 并处理其它功能的状态
                        imageView.setSelected(beautyData.getInt(ivTag, 0) != 0);
                    }
                }
                view.setActivated(true);
                mBeautyDensity.setVisibility(View.VISIBLE);
                mBeautyDensity.setProgress(beautyData.getInt(tag, 0));
            }
        };

        for (ImageView imageView : mBeautyFuncList){
            imageView.setOnClickListener(beautyFuncListener);
        }

        mGlasses.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                glassesFilter.switchAvailable();
                mGlasses.setSelected(glassesFilter.getAvailable());
            }
        });

        mBeautyDensity.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                // 当SeekBar数值改变时调用
                for (ImageView imageView : mBeautyFuncList){
                    if (imageView.isActivated()){
                        // 获取当前被选中的妆容
                        String tag = (String) imageView.getTag();

                        editor.putInt(tag, mBeautyDensity.getProgress());      // 设置浓度数值
                        editor.apply();             // 并提交

                        switch (tag){
                            case "Lighten":
                                whiteFilter.setWhiteLevel((float) mBeautyDensity.getProgress() / 10);
                                break;
                            case "Smooth":
                                smoothFilter.setSmoothLevel((float) mBeautyDensity.getProgress() / 10);
                                break;
                            case "Thinner":
                                faceLiftFilter.setStrength((float) mBeautyDensity.getProgress() / 10);
                                break;

                            case "Blusher":
                                blusherFilter.setStrength((float) mBeautyDensity.getProgress() / 10);
                                break;
                            case "Eyebrow":
                                eyebrowFilter.setStrength((float) mBeautyDensity.getProgress() / 10);
                                break;
                            case "Lip":
                                lipFilter.setStrength((float) mBeautyDensity.getProgress() / 10);
                                break;

                            case "Pupil":
                                pupilFilter.setStrength((float) mBeautyDensity.getProgress() / 10);
                                break;
                            case "Eyeshadow":
                                eyeShadowFiler.setStrength((float) mBeautyDensity.getProgress() / 10);
                                break;
                            case "Furrow":
                                furrowFilter.setStrength((float) mBeautyDensity.getProgress() / 10);
                                break;
                        }
                    }
                }
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
                // 当用户开始拖动SeekBar时调用
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
                // 当用户停止拖动SeekBar时调用
            }
        });
    }

    private void startTrack() {
        mInferenceResult.reset();
        mImageBufferQueue = new ImageBufferQueue(3, CAMERA_PREVIEW_WIDTH, CAMERA_PREVIEW_HEIGHT);
        mStopInference = false;
        mInferenceThread = new Thread(mInferenceRunnable);
        mInferenceThread.start();
    }

    private Thread mInferenceThread;
    private Runnable mInferenceRunnable = new Runnable() {
        public void run() {
            String paramPath = fileDirPath + "/" + mModelName;
            String paramPathFace = fileDirPath + "/" + mModelNameFace;

            try {
                // FIXME INIT 此处修改输入尺寸
                mInferenceWrapper.initModel(800, 600, INPUT_CHANNEL, paramPath, paramPathFace);
            } catch (Exception e) {
                e.printStackTrace();
                System.exit(1);
            }

        }
    };

    private byte[] captureCameraData(IFilter.Frame frame) {
        int bufferSize = frame.getTextureWidth() * frame.getTextureHeight() * 4; // Assuming RGBA format
        ByteBuffer captureBuffer = ByteBuffer.allocateDirect(bufferSize);
        captureBuffer.order(ByteOrder.nativeOrder());

        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, frame.getFrameBuffer());
        GLES20.glReadPixels(0, 0, frame.getTextureWidth(), frame.getTextureHeight(),
                GLES20.GL_RGBA, GL10.GL_UNSIGNED_BYTE, captureBuffer);

        byte[] byteArray = new byte[bufferSize];
        captureBuffer.get(byteArray);

        // If needed, you can manipulate the byteArray here

        return byteArray;
    }

    private void capture() {
        AsyncTask.execute(new Runnable() {
            @Override
            public void run() {
                aglView.capture(new AGLView.CaptureListener() {
                    @Override
                    public void captured(Bitmap bitmap) {
                        if (bitmap == null) {
                            show.setVisibility(View.GONE);
                            return;
                        }
                        show.setVisibility(View.VISIBLE);
                        show.setImageBitmap(bitmap);
                        try {
                            String title = System.currentTimeMillis() + ".png";
                            String path = Environment.getExternalStorageDirectory().getAbsolutePath() + "/DCIM/Camera/" + title;
                            File f = new File(path);
                            if (f.exists()) {
                                boolean exist = f.delete();
                            }
                            FileOutputStream out = new FileOutputStream(f);
                            bitmap.compress(Bitmap.CompressFormat.PNG, 100, out);
                            out.flush();
                            out.close();
                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    Toast.makeText(MainActivityAGLcamera.this, "保存成功", Toast.LENGTH_SHORT).show();
                                }
                            });
                            ContentValues currentVideoValues = new ContentValues();
                            currentVideoValues.put(MediaStore.Images.Media.TITLE, title);
                            currentVideoValues.put(MediaStore.Images.Media.DISPLAY_NAME, title);
                            currentVideoValues.put(MediaStore.Images.Media.DATE_TAKEN, System.currentTimeMillis());
                            currentVideoValues.put(MediaStore.Images.Media.MIME_TYPE, "image/png");
                            currentVideoValues.put(MediaStore.Images.Media.DATA, path);
                            currentVideoValues.put(MediaStore.Images.Media.DESCRIPTION, "com.smzh.aglframework");
                            currentVideoValues.put(MediaStore.Images.Media.SIZE, f.length());
                            getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, currentVideoValues);
                        } catch (IOException e) {
                            e.printStackTrace();
                        }
                    }
                });
            }
        });
    }

    @Override
    public void onPreviewFrame(byte[] bytes, Camera camera) {
        Mat yuvMat = new Mat(camera.getParameters().getPreviewSize().height + camera.getParameters().getPreviewSize().height / 2,
                camera.getParameters().getPreviewSize().width, CvType.CV_8UC1);
        yuvMat.put(0, 0, bytes);

        Mat rgbMat = new Mat();
        Imgproc.cvtColor(yuvMat, rgbMat, Imgproc.COLOR_YUV2RGB_NV21);

//        Core.flip(rgbMat, rgbMat, 1);

        // 竖屏用输入矩阵
        Mat rotatedMat = new Mat();
        Core.rotate(rgbMat, rotatedMat, Core.ROTATE_90_CLOCKWISE);   // FIXME INIT 根据输入图片进行旋转，270对应逆时针，90对应顺时针

//        Mat bgrMat = new Mat();
//        Imgproc.cvtColor(rotatedMat, bgrMat, Imgproc.COLOR_RGB2BGR);
//        if (!isPhotoSaved) {
//            Log.d(TAG, "存储一帧图像");
//            Imgcodecs.imwrite("/storage/emulated/0/DCIM/SkinResult/interfacetest.jpg", bgrMat);
//            isPhotoSaved = true;
//        }

        byte[] imageBytes = new byte[rgbMat.rows() * rgbMat.cols() * (int) (rgbMat.elemSize())];
        rotatedMat.get(0, 0, imageBytes);

        if (mInferenceWrapper != null) {
//            if (count % 60 == 0) {
//                Log.d(TAG, "传入模型" + System.currentTimeMillis());
//            }

            InferenceResult.OutputBuffer outputs = mInferenceWrapper.run(imageBytes);
          //YYX 打印数据长度outputs.mGrid0Out
            Log.d(TAG, "第一人[10]" + outputs.mGrid0Out[10] +"第二人[10]" + outputs.mGrid1Out[10]);

            float[] x = new float[110];
            float[] y = new float[110];
            for (int i = 0; i < outputs.mGrid0Out.length; i++) {
//                        Log.e("yyx","data = "+outputs.mGrid0Out[i]);
                if (i % 2 == 0) {
//                            x[i/2] = outputs.mGrid0Out[i]-2*(outputs.mGrid0Out[i] - zhongjian);
                    x[i / 2] = outputs.mGrid0Out[i];
                } else {
                    y[i / 2] = outputs.mGrid0Out[i];
                }
            }

            // 如果脸移动超过阈值，则更新点位
//            if (isFaceMove(x, y)) {
                points = new Point[110];

                for (int i = 0; i < x.length; i++) {
                    points[i] = new Point(x[i], y[i]);
                }

//                Log.e("yyx","更新鼻尖点位: " + points[79]);
//            if (count % 60 == 0) {
//                Log.e("yyx","鼻尖点位: " + points[79]);
//            }

//                count ++;
//            }
        }

        calculateExtra();

        callRenderer();


        // 以下部分利用OpenCV绘制点
        Scalar color = new Scalar(255, 0, 0); // 点的颜色
        for (Point point : points) {
            Imgproc.circle(rotatedMat, point, 4, color, -1); // 绘制实心圆
        }

        // 显示到界面
        mTrackResultBitmap = Bitmap.createBitmap(rotatedMat.cols(), rotatedMat.rows(), Bitmap.Config.RGB_565);    // 用于储存结果bitmap的成员变量
        Utils.matToBitmap(rotatedMat, mTrackResultBitmap);    // 将rotatedMat转为Bitmap存储

        mTrackResultView.setVisibility(View.VISIBLE);
        mTrackResultView.setScaleType(ImageView.ScaleType.FIT_XY);
        mTrackResultView.setImageBitmap(mTrackResultBitmap);  // 显示这张Bitmap
    }

    //取平台版本
    private String getPlatform() {
        String platform = null;
        try {
            Class<?> classType = Class.forName("android.os.SystemProperties");
            Method getMethod = classType.getDeclaredMethod("get", new Class<?>[]{String.class});
            platform = (String) getMethod.invoke(classType, new Object[]{"ro.board.platform"});
        } catch (Exception e) {
            e.printStackTrace();
        }
        return platform;
    }

    private void callRenderer(){
        aglcamera.enableRenderer();
        synchronized (lock){
            Log.e("yyx","Activity请求渲染:");
            aglView.requestRender();
        }
    }


    /** 计算脸颊、眉心四个点 */
    private void calculateExtra() {
        if (points != null && points.length > 0) {
            points[106] = new Point((points[13].x + points[76].x) / 2, (points[13].y + points[76].y) / 2);
            points[107] = new Point((points[29].x + points[82].x) / 2, (points[29].y + points[82].y) / 2);
            points[108] = new Point((points[45].x + points[49].x) / 2, (points[45].y + points[49].y) / 2);
            points[109] = new Point((points[99].x + points[104].x) / 2, (points[99].y + points[104].y) / 2);
        }
    }

    private boolean isFaceMove(float[] x, float[] y){
        // point未初始化时直接返回true
        if (points != null){
            // 找出关键的五个点
            float leftPupilCenterX = x[38];
            float rightPupilCenterX = x[88];
            float noseTopX = x[80];
            float leftLipTopX = x[52];
            float rightLipTopX = x[61];

            float leftPupilCenterY = y[38];
            float rightPupilCenterY = y[88];
            float noseTopY = y[80];
            float leftLipTopY = y[52];
            float rightLipTopY = y[61];

            // 位移阈值，超过这个值会被判断为移动了
            int threshold = 1;

            if (Math.sqrt(Math.pow((points[38].x - leftPupilCenterX), 2) + Math.pow((points[38].y - leftPupilCenterY), 2)) > threshold){
                return true;
            }
            if (Math.sqrt(Math.pow((points[88].x - rightPupilCenterX), 2) + Math.pow((points[88].y - rightPupilCenterY), 2)) > threshold){
                return true;
            }
            if (Math.sqrt(Math.pow((points[80].x - noseTopX), 2) + Math.pow((points[80].y - noseTopY), 2)) > threshold){
                return true;
            }
            if (Math.sqrt(Math.pow((points[52].x - leftLipTopX), 2) + Math.pow((points[52].y - leftLipTopY), 2)) > threshold){
                return true;
            }
            if (Math.sqrt(Math.pow((points[61].x - rightLipTopX), 2) + Math.pow((points[61].y - rightLipTopY), 2)) > threshold){
                return true;
            }

            return false;
        } else {
            return true;
        }
    }

}