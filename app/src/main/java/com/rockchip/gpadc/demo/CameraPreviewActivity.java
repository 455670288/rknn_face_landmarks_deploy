package com.rockchip.gpadc.demo;

import android.app.Activity;
import android.content.SharedPreferences;
import android.content.res.AssetManager;
import android.content.res.ColorStateList;
import android.content.res.Resources;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.PorterDuffXfermode;
import android.graphics.SurfaceTexture;
import android.hardware.Camera;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.support.v4.content.ContextCompat;
import android.util.Log;
import android.view.Gravity;
import android.view.MotionEvent;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.SeekBar;
import android.widget.TextView;
import android.widget.Toast;

import com.rockchip.gpadc.demo.yolo.InferenceWrapper;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Method;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

import static com.rockchip.gpadc.demo.rga.HALDefine.CAMERA_PREVIEW_HEIGHT;
import static com.rockchip.gpadc.demo.rga.HALDefine.CAMERA_PREVIEW_WIDTH;
import static com.rockchip.gpadc.demo.rga.HALDefine.IM_HAL_TRANSFORM_FLIP_H;
import static com.rockchip.gpadc.demo.yolo.PostProcess.INPUT_CHANNEL;
import static java.lang.Thread.sleep;


public class CameraPreviewActivity extends Activity implements Camera.PreviewCallback {

    private final String TAG = "rkyolo";
    private static final int MAGIC_TEXTURE_ID = 10;

    TSurfaceHolderCallback mSurfaceHolderCallback = null;

    private Camera mCamera0 = null;
    private SurfaceView mSurfaceView = null;
    public SurfaceTexture mSurfaceTexture = null;
    private SurfaceHolder mSurfaceHolder = null;
    public int flip = -1;    // for CAMERA_FACING_BACK(camera comes with RK3588 using this mode),
    // we do not need flip, using -1, or we need using
    // IM_HAL_TRANSFORM_FLIP_H

    private boolean mIsCameraOpened = false;
    private int mCameraId = -1;
    public byte textureBuffer[];

    // for inference
    private String mModelName = "yolov5s.rknn";
    private String platform = "rk3588";
    private InferenceWrapper mInferenceWrapper;
    private String fileDirPath;     // file dir to store model cache
    private ImageBufferQueue mImageBufferQueue;    // intermedia between camera thread and  inference thread
    private InferenceResult mInferenceResult = new InferenceResult();  // detection result
    private int mWidth;    //surface width
    private int mHeight;    //surface height
    private volatile boolean mStopInference = false;

    //draw result
    private TextView mFpsNum1;
    private TextView mFpsNum2;
    private TextView mFpsNum3;
    private TextView mFpsNum4;
    private ImageView mTrackResultView;
    private Bitmap mTrackResultBitmap = null;
    private Canvas mTrackResultCanvas = null;
    private Paint mTrackResultPaint = null;
    private Paint mTrackResultTextPaint = null;

    // func area
    private ImageView mLipStick;
    private ImageView mDelight;
    private ImageView mClose;
    private ImageView mNoColor;
    private ImageView mColor1;
    private ImageView mColor2;
    private ImageView mColor3;
    private ImageView mColor4;
    private LinearLayout mLipStickSetting;
    private LinearLayout mFunctionBar;
    private LinearLayout mFunctionCloseBar;
    /**
     * 进度条 默认3 最大值10
     */
    private SeekBar mColorDensity;
    private int lipStickColorId;

    private boolean isFuncBarShown = false;
    private boolean isColorChosen = false;
    private boolean isDelightOpen = false;

    private PorterDuffXfermode mPorterDuffXfermodeClear;
    private PorterDuffXfermode mPorterDuffXfermodeSRC;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        if (OpenCVLoader.initDebug()) {
            Log.d("yyx OpenCV", "OpenCV successfully loaded");
        } else {
            Log.d("yyx OpenCV", "OpenCV not loaded");
        }
        mFpsNum1 = (TextView) findViewById(R.id.fps_num1);
        mFpsNum2 = (TextView) findViewById(R.id.fps_num2);
        mFpsNum3 = (TextView) findViewById(R.id.fps_num3);
        mFpsNum4 = (TextView) findViewById(R.id.fps_num4);
        mTrackResultView = (ImageView) findViewById(R.id.canvasView);

        fileDirPath = getCacheDir().getAbsolutePath();

        platform = getPlatform();
        Log.d(TAG, "get soc platform:" + platform);

        if (platform.equals("rk3588")) {
//            createFile(mModelName, R.raw.yolov5s_rk3588);
//        } else if (platform.equals("rk356x")) {
//            createFile(mModelName, R.raw.yolov5s_rk3566);
//        } else if (platform.equals("rk3562")) {
//            createFile(mModelName, R.raw.yolov5s_rk3562);
//        } else {
//            Toast toast = Toast.makeText(this, "Can not get platform use RK3588 instead.", Toast.LENGTH_LONG);
//            toast.setGravity(Gravity.CENTER, 0, 0);
//            toast.show();
            createFile(mModelName, R.raw.change_rk3588_unquantized);
        } else if (platform.equals("rk356x")) {
            createFile(mModelName, R.raw.change_rk3566_unquantized);
        }

        try {
            mInferenceResult.init(getAssets());
        } catch (IOException e) {
            e.printStackTrace();
        }

        mInferenceWrapper = new InferenceWrapper();

    }

    @Override
    protected void onDestroy() {
        Log.d(TAG, "onDestroy");

        destroyPreviewView();
        super.onDestroy();
    }

    @Override
    protected void onPause() {
        Log.d(TAG, "onPause");
        stopTrack();
        stopCamera();
        destroyPreviewView();
        super.onPause();

    }

    @Override
    protected void onResume() {
        Log.d(TAG, "onResume");

        initView();
        createPreviewView();
        super.onResume();

    }

    private void initView() {
        mLipStick = (ImageView) findViewById(R.id.func_lip_stick);
        mDelight = (ImageView) findViewById(R.id.func_delight);
        mClose = (ImageView) findViewById(R.id.function_close);
        mNoColor = (ImageView) findViewById(R.id.lip_stick_no_color);

        mColor1 = (ImageView) findViewById(R.id.lip_stick_color_1);
        mColor1.setTag(R.array.dior_999);
        mColor2 = (ImageView) findViewById(R.id.lip_stick_color_2);
        mColor2.setTag(R.array.tf_16);
        //获取id为lip_stick_color_3的ImageView控件
        mColor3 = (ImageView) findViewById(R.id.lip_stick_color_3);
        //设置ImageView控件的标签为R.array.dior_649
        mColor3.setTag(R.array.dior_649);
        mColor4 = (ImageView) findViewById(R.id.lip_stick_color_4);
        mColor4.setTag(R.array.givenchy_16);
        final List<ImageView> lipStickColors = Arrays.asList(mColor1, mColor2, mColor3, mColor4);

        mLipStickSetting = (LinearLayout) findViewById(R.id.lip_stick_setting);
        mFunctionBar = (LinearLayout) findViewById(R.id.function_bar);
        mFunctionCloseBar = (LinearLayout) findViewById(R.id.function_close_bar);

        mColorDensity = (SeekBar) findViewById(R.id.lip_stick_density);
        //设置进度条颜色
        mColorDensity.setThumbTintList(ColorStateList.valueOf(ContextCompat.getColor(this, R.color.white)));
        mColorDensity.setProgressTintList(ColorStateList.valueOf(ContextCompat.getColor(this, R.color.white)));

        mLipStick.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (!isFuncBarShown) {
                    mLipStickSetting.setVisibility(View.VISIBLE);
                    mFunctionBar.setVisibility(View.GONE);
                    mFunctionCloseBar.setVisibility(View.VISIBLE);
                    isFuncBarShown = true;
                }
            }
        });

        mClose.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (isFuncBarShown) {
                    mLipStickSetting.setVisibility(View.GONE);
                    mFunctionBar.setVisibility(View.VISIBLE);
                    mFunctionCloseBar.setVisibility(View.GONE);
                    isFuncBarShown = false;
                }
            }
        });

        mDelight.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (isDelightOpen) {
                    mDelight.setImageDrawable(getDrawable(R.drawable.icon_delight));
                    isDelightOpen = false;
                } else {
                    mDelight.setImageDrawable(getDrawable(R.drawable.icon_delight_off));
                    isDelightOpen = true;
                }
            }
        });


        mNoColor.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (isColorChosen) {
                    mColorDensity.setVisibility(View.GONE);
                    isColorChosen = false;

                    for (ImageView color : lipStickColors) {
                        color.setSelected(false);
                    }
                }
            }
        });

        View.OnClickListener lipStickFuncListener = new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                lipStickColorId = (int) view.getTag();

                view.setSelected(true);
                for (ImageView color : lipStickColors) {
                    if ((int) color.getTag() != lipStickColorId) {
                        color.setSelected(false);
                    }
                }

                if (!isColorChosen) {
                    mColorDensity.setVisibility(View.VISIBLE);
                    isColorChosen = true;
                }
            }
        };

        mColor1.setOnClickListener(lipStickFuncListener);
        mColor2.setOnClickListener(lipStickFuncListener);
        mColor3.setOnClickListener(lipStickFuncListener);
        mColor4.setOnClickListener(lipStickFuncListener);
    }

    private boolean createPreviewView() {
        mSurfaceView = findViewById(R.id.surfaceViewCamera1);
        mSurfaceHolder = mSurfaceView.getHolder();
//        mSurfaceView.setZOrderMediaOverlay(true);

        mSurfaceTexture = new SurfaceTexture(MAGIC_TEXTURE_ID);

        mSurfaceHolderCallback = new TSurfaceHolderCallback();
        mSurfaceHolder.addCallback(mSurfaceHolderCallback);

        return true;
    }

    private void destroyPreviewView() {
        if (mSurfaceHolder != null) {
            mSurfaceHolder.removeCallback(mSurfaceHolderCallback);
            mSurfaceHolderCallback = null;
            mSurfaceHolder = null;
        }

    }

    public Bitmap convertByteArrayToBitmap(byte[] byteArray, int width, int height) {
        Mat mat = new Mat(width, height, CvType.CV_8UC3);
        mat.put(0, 0, byteArray);
        Mat bgrMat = new Mat();
        Imgproc.cvtColor(mat, bgrMat, Imgproc.COLOR_RGB2BGR);
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(bgrMat, bitmap);
        return bitmap;
    }

    @Override
    public void onPreviewFrame(byte[] data, Camera camera) {
        if (data == null) {
            return;
        }
        mCamera0.addCallbackBuffer(data);
        ImageBufferQueue.ImageBuffer imageBuffer = mImageBufferQueue.getFreeBuffer();

        if (imageBuffer != null) {

            // RK_FORMAT_YCrCb_420_SP -> RK_FORMAT_RGBA_8888
            // flip for CAMERA_FACING_FRONT
//            byte[] imageRGBA_8888 = new byte[CAMERA_PREVIEW_WIDTH*CAMERA_PREVIEW_HEIGHT*4];
//            RGA.colorConvertAndFlip(data, RK_FORMAT_YCrCb_420_SP,
//                    imageRGBA_8888, RK_FORMAT_RGBA_8888,
//                    CAMERA_PREVIEW_WIDTH, CAMERA_PREVIEW_HEIGHT, this.flip);
//            Log.e("yyx","height = "+camera.getParameters().getPreviewSize().height+"--width = "+camera.getParameters().getPreviewSize().width);
//            Mat rgbaMat = new Mat(CAMERA_PREVIEW_HEIGHT, CAMERA_PREVIEW_WIDTH, CvType.CV_8UC4);
//            rgbaMat.put(0, 0, imageRGBA_8888);
//            // 创建一个新的 Mat 对象来存储 RGB 格式的数据
//            Mat rgbMat = new Mat(CAMERA_PREVIEW_HEIGHT, CAMERA_PREVIEW_WIDTH, CvType.CV_8UC3);
//
//// 将 RGBA 格式转换为 RGB 格式
//            Imgproc.cvtColor(rgbaMat, rgbMat, Imgproc.COLOR_RGBA2RGB);
//
//// 将 Mat 对象转换为字节数组na
//            rgbMat.get(0, 0, imageBuffer.mImage);

            Mat yuvMat = new Mat(camera.getParameters().getPreviewSize().height + camera.getParameters().getPreviewSize().height / 2,
                    camera.getParameters().getPreviewSize().width, CvType.CV_8UC1);
            yuvMat.put(0, 0, data);

            Mat rgbMat = new Mat();
            Imgproc.cvtColor(yuvMat, rgbMat, Imgproc.COLOR_YUV2RGB_NV21);
            Core.flip(rgbMat, rgbMat, 1);
            byte[] imageBytes = new byte[rgbMat.rows() * rgbMat.cols() * (int) (rgbMat.elemSize())];
            rgbMat.get(0, 0, imageBytes);

            InferenceResult.OutputBuffer outputs = mInferenceWrapper.run(imageBytes);

            mTrackResultBitmap = Bitmap.createBitmap(rgbMat.cols(), rgbMat.rows(), Bitmap.Config.RGB_565);
            Utils.matToBitmap(rgbMat, mTrackResultBitmap);

            Mat mat = new Mat(mTrackResultBitmap.getHeight(), mTrackResultBitmap.getWidth(), CvType.CV_8UC4);
//            float zhongjian = mTrackResultBitmap.getHeight()/2;
            Utils.bitmapToMat(mTrackResultBitmap, mat);

            if (isDelightOpen) {
                Mat vChannel = new Mat();
                Mat hsvImage = new Mat();
                Imgproc.cvtColor(mat, hsvImage, Imgproc.COLOR_RGB2HSV);
                Core.extractChannel(hsvImage, vChannel, 2);
                double lightGamma = Core.mean(vChannel).val[0] / 138;
                mat = gammaCorrection(mat, lightGamma);
            }

            if (isColorChosen) {
                Point[] points = new Point[106];

                float[] x = new float[106];
                float[] y = new float[106];
                for (int i = 0; i < outputs.mGrid0Out.length; i++) {
//                        Log.e("yyx","data = "+outputs.mGrid0Out[i]);
                    if (i % 2 == 0) {
//                            x[i/2] = outputs.mGrid0Out[i]-2*(outputs.mGrid0Out[i] - zhongjian);
                        x[i / 2] = outputs.mGrid0Out[i];
                    } else {
                        y[i / 2] = outputs.mGrid0Out[i];
                    }
                }
                for (int i = 0; i < x.length; i++) {
                    points[i] = new Point(x[i], y[i]);
                }

                int[] lipStick = getResources().getIntArray(lipStickColorId);
                double alpha = (double) mColorDensity.getProgress() / 10;
                // 创建一个Scalar对象，参数为lipStick数组中0、1、2、3个元素的值，alpha为透明度
                Scalar color = new Scalar(lipStick[0], lipStick[1], lipStick[2], 255 * alpha);

//            Scalar color = new Scalar(255, 0, 0); // 点的颜色，这里使用蓝色
                for (Point point : points) {
                    Imgproc.circle(mat, point, 4, color, -1); // 绘制实心圆
                }

                int[] upperLip = {52, 66, 62, 70, 61, 68, 67, 71, 63, 64};
                int[] lowerLip = {52, 65, 54, 60, 57, 69, 61, 58, 59, 53, 56, 55};
                Point[] upperLipMap = new Point[10];
                Point[] lowerLipMap = new Point[12];

                for (int i = 0; i < 12; i++) {
                    if (i < 10) {
                        upperLipMap[i] = points[upperLip[i]];
                    }
                    lowerLipMap[i] = points[lowerLip[i]];
                }

                List<MatOfPoint> upperLipMat = new ArrayList<>();
                List<MatOfPoint> lowerLipMat = new ArrayList<>();
                upperLipMat.add(new MatOfPoint(upperLipMap));
                lowerLipMat.add(new MatOfPoint(lowerLipMap));

                Mat mask = new Mat(mTrackResultBitmap.getHeight(), mTrackResultBitmap.getWidth(), CvType.CV_8UC4, new Scalar(255, 255, 255, 255));

                Imgproc.fillPoly(mask, upperLipMat, color);
                Imgproc.fillPoly(mask, lowerLipMat, color);

                Mat merge = mat.clone();
                Core.min(mat, mask, merge);
                Core.addWeighted(mat, (1 - alpha), merge, alpha, 0, mat);
            }

            if (true) {
                AssetManager assetManager = getAssets();
                try {
                    // 读取图像文件
                    String filePath = "eye1.png";
                    InputStream inputStream = assetManager.open(filePath);
                    // 将输入流转换为Bitmap
                    Bitmap bitmap = BitmapFactory.decodeStream(inputStream);
                    // 将Bitmap转换为Mat
                    Mat mat1 = new Mat();
                    Utils.bitmapToMat(bitmap, mat1);

                    inputStream.close();
                } catch (IOException e) {
                    e.printStackTrace();
                }

            }


            Utils.matToBitmap(mat, mTrackResultBitmap);
            mTrackResultView.setScaleType(ImageView.ScaleType.FIT_XY);
            mTrackResultView.setImageBitmap(mTrackResultBitmap);

            mat.release();
            rgbMat.release();
            yuvMat.release();

            mImageBufferQueue.postBuffer(imageBuffer);
        }
    }

    private class TSurfaceHolderCallback implements SurfaceHolder.Callback {

        @Override
        public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
            Log.d(TAG, "surfaceChanged");
            mWidth = width;
            mHeight = height;

            textureBuffer = new byte[CAMERA_PREVIEW_WIDTH * CAMERA_PREVIEW_HEIGHT * 4];
        }

        @Override
        public void surfaceCreated(SurfaceHolder holder) {
            Log.d(TAG, "surfaceCreated");

            startCamera();
            startTrack();

        }

        @Override
        public void surfaceDestroyed(SurfaceHolder holder) {
            Log.d(TAG, "surfaceDestroyed");

            stopTrack();
            stopCamera();
        }
    }

    private boolean startCamera() {
        if (mIsCameraOpened) {
            return true;
        }

        //(Camera.CameraInfo.CAMERA_FACING_BACK);
        int num = Camera.getNumberOfCameras();
        if (num > 2)
            mCameraId = 2;
        else
            mCameraId = 0;
        Log.d(TAG, "mCameraId = " + mCameraId);
        Camera.CameraInfo camInfo = new Camera.CameraInfo();
        try {
            Camera.getCameraInfo(mCameraId, camInfo);
            if (mCameraId != -1) {
                mCamera0 = Camera.open(mCameraId);
            } else {
                mCamera0 = Camera.open();
            }
            Log.d(TAG, "mCamera0 = " + mCamera0);
            Log.d(TAG, "camera facing: " + camInfo.facing);
            if (Camera.CameraInfo.CAMERA_FACING_FRONT == camInfo.facing) {
                this.flip = IM_HAL_TRANSFORM_FLIP_H;
            }

        } catch (RuntimeException e) {
            Log.w(TAG, "Unable to open camera!");
            Toast toast = Toast.makeText(this, "Unable to open camera!", Toast.LENGTH_LONG);
            toast.setGravity(Gravity.CENTER, 0, 0);
            toast.show();
            return false;
        }

        setCameraParameters();

        try {
            mCamera0.setPreviewDisplay(mSurfaceHolder);
            mCamera0.setDisplayOrientation(0);
            int BUFFER_SIZE0 = CAMERA_PREVIEW_WIDTH * CAMERA_PREVIEW_HEIGHT * 3 / 2; // NV21
            byte[][] mPreviewData0 = new byte[][]{new byte[BUFFER_SIZE0], new byte[BUFFER_SIZE0], new byte[BUFFER_SIZE0]};
            //================================
            for (byte[] buffer : mPreviewData0)
                mCamera0.addCallbackBuffer(buffer);
            mCamera0.setPreviewCallbackWithBuffer(this);

            //==================================
            mCamera0.startPreview();
        } catch (Exception e) {
            mCamera0.release();
            return false;
        }

        mIsCameraOpened = true;

        return true;
    }

    private void stopCamera() {
        if (mIsCameraOpened) {
            mCamera0.setPreviewCallback(null);
            mCamera0.stopPreview();
            mCamera0.release();
            mCamera0 = null;
            mIsCameraOpened = false;
        }

    }

    private void setCameraParameters() {
        Camera.Parameters parameters;
        boolean checkWH = false;
        parameters = mCamera0.getParameters();
        int nearest_width_index = 0;
        int nearest_width_value = 1920;

        List<Camera.Size> sizes = parameters.getSupportedPreviewSizes();
        for (int i = 0; i < sizes.size(); i++) {
            Camera.Size size = sizes.get(i);

            if (Math.abs(size.width - CAMERA_PREVIEW_WIDTH) < nearest_width_value) {
                nearest_width_value = Math.abs(size.width - CAMERA_PREVIEW_WIDTH);
                nearest_width_index = i;
            }

            if ((size.width == CAMERA_PREVIEW_WIDTH) && (size.height == CAMERA_PREVIEW_HEIGHT)) {
                checkWH = true;
            }

            Log.v(TAG, "Camera Supported Preview Size = " + size.width + "x" + size.height);
        }
        if (!checkWH) {
            Log.e(TAG, "Camera don't support this preview Size = " + CAMERA_PREVIEW_WIDTH + "x" + CAMERA_PREVIEW_HEIGHT);
            CAMERA_PREVIEW_WIDTH = sizes.get(nearest_width_index).width;
            CAMERA_PREVIEW_HEIGHT = sizes.get(nearest_width_index).height;
        }

        Log.v(TAG, "Use preview Size = " + CAMERA_PREVIEW_WIDTH + "x" + CAMERA_PREVIEW_HEIGHT);

        parameters.setPreviewSize(CAMERA_PREVIEW_WIDTH, CAMERA_PREVIEW_HEIGHT);

        if (parameters.isZoomSupported()) {
            parameters.setZoom(0);
        }
        mCamera0.setParameters(parameters);
        Log.i(TAG, "mCamera0 set parameters success.");
    }

    private void startTrack() {
        mInferenceResult.reset();
        mImageBufferQueue = new ImageBufferQueue(3, CAMERA_PREVIEW_WIDTH, CAMERA_PREVIEW_HEIGHT);
        mStopInference = false;
        mInferenceThread = new Thread(mInferenceRunnable);
        mInferenceThread.start();
    }

    private void stopTrack() {

        mStopInference = true;
        try {
            if (mInferenceThread != null) {
                mInferenceThread.join();
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        if (mImageBufferQueue != null) {
            mImageBufferQueue.release();
            mImageBufferQueue = null;
        }
    }

    private Thread mInferenceThread;
    private Runnable mInferenceRunnable = new Runnable() {
        public void run() {

            int count = 0;
            long oldTime = System.currentTimeMillis();
            long currentTime;

            updateMainUI(1, 0);

            String paramPath = fileDirPath + "/" + mModelName;

            try {
                mInferenceWrapper.initModel(CAMERA_PREVIEW_HEIGHT, CAMERA_PREVIEW_WIDTH, INPUT_CHANNEL, paramPath,"");
            } catch (Exception e) {
                e.printStackTrace();
                System.exit(1);
            }

            mStopInference = true;
            while (!mStopInference) {
                ImageBufferQueue.ImageBuffer buffer = mImageBufferQueue.getReadyBuffer();

                if (buffer == null) {
                    try {
//                        Log.w(TAG, "buffer is null.");
                        sleep(10);
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                    continue;
                }

                InferenceResult.OutputBuffer outputs = mInferenceWrapper.run(buffer.mImage);
                mInferenceResult.setResult(outputs);

                mImageBufferQueue.releaseBuffer(buffer);
                Log.d("yd", "current fps = ");
                if (++count >= 30) {
                    currentTime = System.currentTimeMillis();

                    float fps = count * 1000.f / (currentTime - oldTime);

                    oldTime = currentTime;
                    count = 0;
                    updateMainUI(0, fps);

                }

//                updateMainUI(1, 0);
            }

//            mInferenceWrapper.deinit();
        }
    };

    @Override
    public boolean onTouchEvent(MotionEvent event) {

        Log.e("yyx", "onTouchEvent = " + event.getAction());
        if (event.getAction() == MotionEvent.ACTION_DOWN) {
            if (!mStopInference) {
                mStopInference = true;
            } else {
                startTrack();
            }
        }
        return super.onTouchEvent(event);
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

            if (!file.exists() || isFirstRun()) {

                InputStream ins = getResources().openRawResource(id);// 通过raw得到数据资源
                FileOutputStream fos = new FileOutputStream(file);
                byte[] buffer = new byte[8192];
                int count = 0;

                while ((count = ins.read(buffer)) > 0) {
                    fos.write(buffer, 0, count);
                }

                fos.close();
                ins.close();

                Log.d(TAG, "Create " + filePath);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private boolean isFirstRun() {
        SharedPreferences sharedPreferences = getSharedPreferences("setting", MODE_PRIVATE);
        boolean isFirstRun = sharedPreferences.getBoolean("isFirstRun", true);
        SharedPreferences.Editor editor = sharedPreferences.edit();
        if (isFirstRun) {
            editor.putBoolean("isFirstRun", false);
            editor.commit();
        }

        return isFirstRun;
    }

    // UI线程，用于更新处理结果
    private Handler mHandler = new Handler() {
        @Override
        public void handleMessage(Message msg) {
            if (msg.what == 0) {
                float fps = (float) msg.obj;

                DecimalFormat decimalFormat = new DecimalFormat("00.00");
                String fpsStr = decimalFormat.format(fps);
                mFpsNum1.setText(String.valueOf(fpsStr.charAt(0)));
                mFpsNum2.setText(String.valueOf(fpsStr.charAt(1)));
                mFpsNum3.setText(String.valueOf(fpsStr.charAt(3)));
                mFpsNum4.setText(String.valueOf(fpsStr.charAt(4)));
            } else {
//                showTrackSelectResults();
            }
        }
    };

    private void updateMainUI(int type, Object data) {
        Message msg = mHandler.obtainMessage();
        msg.what = type;
        msg.obj = data;
        mHandler.sendMessage(msg);
    }

    public static int sp2px(float spValue) {
        Resources r = Resources.getSystem();
        final float scale = r.getDisplayMetrics().scaledDensity;
        return (int) (spValue * scale + 0.5f);
    }

    private void showTrackSelectResults() {

        int width = CAMERA_PREVIEW_WIDTH;
        int height = CAMERA_PREVIEW_HEIGHT;
        float[] result = mInferenceResult.getResultNew();
        if (result == null) {
            return;
        }
        for (int i = 0; i < result.length; i++) {
            Log.e("yyx", "result[" + i + "]" + "=" + result[i]);
        }
//        if (mTrackResultBitmap == null) {

        mTrackResultBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
//            mTrackResultCanvas = new Canvas(mTrackResultBitmap);
//
//            //用于画线
//            mTrackResultPaint = new Paint();
//            mTrackResultPaint.setColor(0xff06ebff);
//            mTrackResultPaint.setStrokeJoin(Paint.Join.ROUND);
//            mTrackResultPaint.setStrokeCap(Paint.Cap.ROUND);
//            mTrackResultPaint.setStrokeWidth(4);
//            mTrackResultPaint.setStyle(Paint.Style.STROKE);
//            mTrackResultPaint.setTextAlign(Paint.Align.LEFT);
//            mTrackResultPaint.setTextSize(sp2px(10));
//            mTrackResultPaint.setTypeface(Typeface.SANS_SERIF);
//            mTrackResultPaint.setFakeBoldText(false);
//
//            //用于文字
//            mTrackResultTextPaint = new Paint();
//            mTrackResultTextPaint.setColor(0xff06ebff);
//            mTrackResultTextPaint.setStrokeWidth(2);
//            mTrackResultTextPaint.setTextAlign(Paint.Align.LEFT);
//            mTrackResultTextPaint.setTextSize(sp2px(12));
//            mTrackResultTextPaint.setTypeface(Typeface.SANS_SERIF);
//            mTrackResultTextPaint.setFakeBoldText(false);
//
//
//            mPorterDuffXfermodeClear = new PorterDuffXfermode(PorterDuff.Mode.CLEAR);
//            mPorterDuffXfermodeSRC = new PorterDuffXfermode(PorterDuff.Mode.SRC);
//        }
//
//        // clear canvas
//        mTrackResultPaint.setXfermode(mPorterDuffXfermodeClear);
//        mTrackResultCanvas.drawPaint(mTrackResultPaint);
//        mTrackResultPaint.setXfermode(mPorterDuffXfermodeSRC);
//
//        //detect result
//        ArrayList<InferenceResult.Recognition> recognitions = mInferenceResult.getResult(mInferenceWrapper);
//        for (int i=0; i<recognitions.size(); ++i) {
//            InferenceResult.Recognition rego = recognitions.get(i);
//            RectF detection = rego.getLocation();
//
//            detection.left *= width;
//            detection.right *= width;
//            detection.top *= height;
//            detection.bottom *= height;
//
////            Log.d(TAG, rego.toString());
////            Log.d(TAG, detection.toString());
//
//            mTrackResultCanvas.drawRect(detection, mTrackResultPaint);
//            mTrackResultCanvas.drawText(rego.getTrackId() + " - " + mInferenceResult.mPostProcess.getLabelTitle(rego.getId()),
//                    detection.left+5, detection.bottom-5, mTrackResultTextPaint);
//            Log.e("yyx","reson = "+rego.getTrackId()+" - "+mInferenceResult.mPostProcess.getLabelTitle(rego.getId()));
//        }
//
        Point[] points = new Point[106];
        Mat mat = new Mat(mTrackResultBitmap.getHeight(), mTrackResultBitmap.getWidth(), CvType.CV_8UC4);
        float zhongjian = mTrackResultBitmap.getHeight() / 2;
        Utils.bitmapToMat(mTrackResultBitmap, mat);
        float[] x = new float[106];
        float[] y = new float[106];
        for (int i = 0; i < result.length; i++) {
//                        Log.e("yyx","data = "+outputs.mGrid0Out[i]);
            if (i % 2 == 0) {
//                            x[i/2] = outputs.mGrid0Out[i]-2*(outputs.mGrid0Out[i] - zhongjian);
                x[i / 2] = result[i];
            } else {
                y[i / 2] = result[i];
            }
        }
        for (int i = 0; i < x.length; i++) {
//                        Log.e("yyx","x["+i+"]="+x[i]);
//                        Log.e("yyx","y["+i+"]="+y[i]);
            points[i] = new Point(x[i], y[i]);
        }
        Scalar color = new Scalar(255, 0, 0); // 点的颜色，这里使用蓝色
        for (Point point : points) {
            Imgproc.circle(mat, point, 4, color, -1); // 绘制实心圆
        }
//        Bitmap resultBitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(mat, mTrackResultBitmap);

        mTrackResultView.setScaleType(ImageView.ScaleType.FIT_XY);
        mTrackResultView.setImageBitmap(mTrackResultBitmap);
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

    /**
     * 对图像进行伽马矫正，使亮度正常
     *
     * @param image 原图
     * @param gamma 伽马值
     */
    public static Mat gammaCorrection(Mat image, double gamma) {
        Mat lookUpTable = new Mat(1, 256, CvType.CV_8U);
        byte[] lookUpTableData = new byte[(int) (lookUpTable.total() * lookUpTable.channels())];

        for (int i = 0; i < lookUpTable.cols(); i++) {
            lookUpTableData[i] = saturate(Math.pow(i / 255.0, gamma) * 255.0);
        }
        lookUpTable.put(0, 0, lookUpTableData);

        Mat img = new Mat();
        Core.LUT(image, lookUpTable, img);

        return img;
    }

    /**
     * Double 转 byte 避免溢出
     */
    private static byte saturate(double val) {
        int iVal = (int) Math.round(val);
        iVal = iVal > 255 ? 255 : (Math.max(iVal, 0));
        return (byte) iVal;
    }
}
