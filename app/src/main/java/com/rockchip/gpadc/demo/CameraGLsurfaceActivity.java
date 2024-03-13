package com.rockchip.gpadc.demo;

import android.graphics.Bitmap;
import android.graphics.SurfaceTexture;
import android.hardware.Camera;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.opengl.Matrix;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.WindowManager;
import android.widget.ImageView;

import com.rockchip.gpadc.demo.utils.LogUtil;
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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

import static android.opengl.GLSurfaceView.RENDERMODE_WHEN_DIRTY;
import static com.rockchip.gpadc.demo.rga.HALDefine.CAMERA_PREVIEW_HEIGHT;
import static com.rockchip.gpadc.demo.rga.HALDefine.CAMERA_PREVIEW_WIDTH;
import static com.rockchip.gpadc.demo.yolo.PostProcess.INPUT_CHANNEL;
import static java.lang.Thread.sleep;

public class CameraGLsurfaceActivity extends AppCompatActivity implements SurfaceTexture.OnFrameAvailableListener{
    public SurfaceTexture mSurfaceTexture;

    public static Camera camera;
    private int camera_status = 120;
    GLSurfaceView mCameraGlsurfaceView;
    public MyRender mRenderer;

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
    private ImageView mTrackResultView;
    private Bitmap mTrackResultBitmap = null;
    private Point[] points;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);
        setContentView(R.layout.activity_camera_glsurface);

        if (OpenCVLoader.initDebug()) {
            Log.d("yyx OpenCV", "OpenCV successfully loaded");
        } else {
            Log.d("yyx OpenCV", "OpenCV not loaded");
        }
        fileDirPath = getCacheDir().getAbsolutePath();
        createFile(mModelName, R.raw.change_rk3588_unquantized);
        try {
            mInferenceResult.init(getAssets());
        } catch (IOException e) {
            e.printStackTrace();
        }
        mInferenceWrapper = new InferenceWrapper();

        initView();
        mCameraGlsurfaceView.setEGLContextClientVersion(2);//在setRenderer()方法前调用此方法
        mRenderer = new MyRender();
        mCameraGlsurfaceView.setRenderer(mRenderer);
        mCameraGlsurfaceView.setRenderMode(RENDERMODE_WHEN_DIRTY);
    }
    private void initView(){
        mCameraGlsurfaceView = findViewById(R.id.camera_glsurface_view);
        mTrackResultView = (ImageView) findViewById(R.id.canvasView);
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
    @Override
    public void onFrameAvailable(SurfaceTexture surfaceTexture) {
        mCameraGlsurfaceView.requestRender();
    }

    public class MyRender implements GLSurfaceView.Renderer {
        private final String vertexShaderCode = "uniform mat4 textureTransform;\n" +
                "attribute vec2 inputTextureCoordinate;\n" +
                "attribute vec4 position;            \n" +//NDK坐标点
                "varying   vec2 textureCoordinate; \n" +//纹理坐标点变换后输出
                "\n" +
                " void main() {\n" +
                "     gl_Position = textureTransform*position;\n" +
                "     textureCoordinate = inputTextureCoordinate;\n" +
                " }";

        private final String fragmentShaderCode = "#extension GL_OES_EGL_image_external : require\n" +
                "precision mediump float;\n" +
                "uniform samplerExternalOES videoTex;\n" +
                "varying vec2 textureCoordinate;\n" +
                "\n" +
                "void main() {\n" +
                "    vec4 tc = texture2D(videoTex, textureCoordinate);\n" +
//                "    float color = tc.r * 0.3 + tc.g * 0.59 + tc.b * 0.11;\n" +  //所有视图修改成黑白
//                "    gl_FragColor = vec4(color,color,color,1.0);\n" +
                "    gl_FragColor = vec4(tc.r,tc.g,tc.b,1.0);\n" +//原色
                "}\n";

        private FloatBuffer mPosBuffer;
        private FloatBuffer mTexBuffer;
//        private float[] mPosCoordinate = {-1, -1, -1, 1, 1, -1, 1, 1};
        private float[] mPosCoordinate = {-1, 1, 1, 1, -1, -1, 1, -1};
        private float[] mTexCoordinateBackRight = {1, 1, 0, 1, 1, 0, 0, 0};//顺时针转90并沿Y轴翻转  后摄像头正确，前摄像头上下颠倒
//        private float[] mTexCoordinateForntRight = {0, 1, 1, 1, 0, 0, 1, 0};//顺时针旋转90  后摄像头上下颠倒了，前摄像头正确
        private float[] mTexCoordinateForntRight = {0, 0, 1, 0, 0, 1, 1, 1};//顺时针旋转90  后摄像头上下颠倒了，前摄像头正确

        public int mProgram;
        public boolean mBoolean = false;

        public MyRender() {
            Matrix.setIdentityM(mProjectMatrix, 0);
            Matrix.setIdentityM(mCameraMatrix, 0);
            Matrix.setIdentityM(mMVPMatrix, 0);
            Matrix.setIdentityM(mTempMatrix, 0);
        }

        private int loadShader(int type, String shaderCode) {
            int shader = GLES20.glCreateShader(type);
            // 添加上面编写的着色器代码并编译它
            GLES20.glShaderSource(shader, shaderCode);
            GLES20.glCompileShader(shader);
            return shader;
        }

        private void creatProgram() {
            //通常做法
//            String vertexSource = AssetsUtils.read(CameraGlSurfaceShowActivity.this, "vertex_texture.glsl");
//            int vertexShader = loadShader(GLES20.GL_VERTEX_SHADER, vertexSource);
//            String fragmentSource = AssetsUtils.read(CameraGlSurfaceShowActivity.this, "fragment_texture.glsl");
//            int fragmentShader = loadShader(GLES20.GL_FRAGMENT_SHADER, fragmentSource);
            int vertexShader = loadShader(GLES20.GL_VERTEX_SHADER, vertexShaderCode);
            int fragmentShader = loadShader(GLES20.GL_FRAGMENT_SHADER, fragmentShaderCode);
            // 创建空的OpenGL ES程序
            mProgram = GLES20.glCreateProgram();

            // 添加顶点着色器到程序中
            GLES20.glAttachShader(mProgram, vertexShader);

            // 添加片段着色器到程序中
            GLES20.glAttachShader(mProgram, fragmentShader);

            // 创建OpenGL ES程序可执行文件
            GLES20.glLinkProgram(mProgram);

            // 释放shader资源
            GLES20.glDeleteShader(vertexShader);
            GLES20.glDeleteShader(fragmentShader);
        }


        private FloatBuffer convertToFloatBuffer(float[] buffer) {
            FloatBuffer fb = ByteBuffer.allocateDirect(buffer.length * 4)
                    .order(ByteOrder.nativeOrder())
                    .asFloatBuffer();
            fb.put(buffer);
            fb.position(0);
            return fb;
        }

        private int uPosHandle;
        private int aTexHandle;
        private int mMVPMatrixHandle;
        private float[] mProjectMatrix = new float[16];
        private float[] mCameraMatrix = new float[16];
        private float[] mMVPMatrix = new float[16];
        private float[] mTempMatrix = new float[16];

        //添加程序到ES环境中
        private void activeProgram() {
            // 将程序添加到OpenGL ES环境
            GLES20.glUseProgram(mProgram);

            mSurfaceTexture.setOnFrameAvailableListener(CameraGLsurfaceActivity.this);
            // 获取顶点着色器的位置的句柄
            uPosHandle = GLES20.glGetAttribLocation(mProgram, "position");
            aTexHandle = GLES20.glGetAttribLocation(mProgram, "inputTextureCoordinate");
            mMVPMatrixHandle = GLES20.glGetUniformLocation(mProgram, "textureTransform");

            mPosBuffer = convertToFloatBuffer(mPosCoordinate);

            if(camera_status == 0){
                mTexBuffer = convertToFloatBuffer(mTexCoordinateBackRight);
            }else{
                mTexBuffer = convertToFloatBuffer(mTexCoordinateForntRight);
            }

            GLES20.glVertexAttribPointer(uPosHandle, 2, GLES20.GL_FLOAT, false, 0, mPosBuffer);
            GLES20.glVertexAttribPointer(aTexHandle, 2, GLES20.GL_FLOAT, false, 0, mTexBuffer);

            // 启用顶点位置的句柄
            GLES20.glEnableVertexAttribArray(uPosHandle);
            GLES20.glEnableVertexAttribArray(aTexHandle);
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

                int count = 0;
                long oldTime = System.currentTimeMillis();
                long currentTime;

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

                    }

                }
            }
        };
        @Override
        public void onSurfaceCreated(GL10 gl, EGLConfig config) {
            GLES20.glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            mSurfaceTexture = new SurfaceTexture(createOESTextureObject());
            creatProgram();
            startTrack();
//            mProgram = ShaderUtils.createProgram(CameraGlSurfaceShowActivity.this, "vertex_texture.glsl", "fragment_texture.glsl");

//            camera = Camera.open(camera_status);注释1
            int mCameraId;

            int num = Camera.getNumberOfCameras();
            if (num > 2)
                mCameraId = 2;
            else
                mCameraId = 0;
            String TAG = "uu";
            Log.d(TAG,"mCameraId = " + mCameraId);
            Camera.CameraInfo camInfo = new Camera.CameraInfo();
            try {
                Camera.getCameraInfo(mCameraId, camInfo);
                if (mCameraId != -1) {
                    camera = Camera.open(mCameraId);
                } else {
                    camera = Camera.open();
                }
                Log.d(TAG, "camera = " + camera);
                Log.d(TAG, "camera facing: " + camInfo.facing);
            } catch (RuntimeException e) {
                Log.w(TAG, "Unable to open camera!");
            }

            try {
                camera.setPreviewTexture(mSurfaceTexture);
                camera.startPreview();
            } catch (IOException e) {
                e.printStackTrace();
            }
            activeProgram();
            camera.setPreviewCallback(new Camera.PreviewCallback() {
                @Override
                public void onPreviewFrame(byte[] data, Camera cameraT) {
                    if (data == null) {
                        return;
                    }
                    camera.addCallbackBuffer(data);
                    ImageBufferQueue.ImageBuffer imageBuffer = mImageBufferQueue.getFreeBuffer();

                    if (imageBuffer != null) {
                        Mat yuvMat = new Mat(cameraT.getParameters().getPreviewSize().height + cameraT.getParameters().getPreviewSize().height / 2,
                                cameraT.getParameters().getPreviewSize().width, CvType.CV_8UC1);
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
                        if (true) {
                            points = new Point[106];

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

                            int[] lipStick = getResources().getIntArray(R.array.dior_999);
                            double alpha = (double) 10 / 10;
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

                        Utils.matToBitmap(mat, mTrackResultBitmap);
                        mTrackResultView.setScaleType(ImageView.ScaleType.FIT_XY);
                        mTrackResultView.setImageBitmap(mTrackResultBitmap);

                        mat.release();
                        rgbMat.release();
                        yuvMat.release();

                        mImageBufferQueue.postBuffer(imageBuffer);
                    }
                }
            });

        }

        @Override
        public void onSurfaceChanged(GL10 gl, int width, int height) {
            LogUtil.d("TAG", "onSurfaceChanged: "+width+" "+height);
            GLES20.glViewport(0, 0, width, height);
//            Matrix.scaleM(mMVPMatrix,0,1,-1,1);
            float ratio = (float) width / height;
           /* Matrix.orthoM(mProjectMatrix, 0, -1, 1, -ratio, ratio, 1, 7);// 3和7代表远近视点与眼睛的距离，非坐标点
            Matrix.setLookAtM(mCameraMatrix, 0, 0, 0, 3, 0f, 0f, 0f, 0f, 1.0f, 0.0f);// 3代表眼睛的坐标点*/


             Matrix.orthoM(mProjectMatrix,0,-1,1,-ratio, ratio, 1,3);
             Matrix.setLookAtM(mCameraMatrix,0,0,0,1,0,0,0,0,1,0);

            Matrix.multiplyMM(mMVPMatrix, 0, mProjectMatrix, 0, mCameraMatrix, 0);
        }

        @Override
        public void onDrawFrame(GL10 gl) {
            if(mBoolean){
                activeProgram();
                mBoolean = false;
            }
            if (mSurfaceTexture != null) {
                GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT | GLES20.GL_DEPTH_BUFFER_BIT);
                mSurfaceTexture.updateTexImage();
                GLES20.glUniformMatrix4fv(mMVPMatrixHandle, 1, false, mMVPMatrix, 0);
                GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, mPosCoordinate.length / 2);
            }
        }
    }
    public static int createOESTextureObject() {
        int[] tex = new int[1];
        //生成一个纹理
        GLES20.glGenTextures(1, tex, 0);
        //将此纹理绑定到外部纹理上
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, tex[0]);
        //设置纹理过滤参数
        GLES20.glTexParameterf(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                GL10.GL_TEXTURE_MIN_FILTER, GL10.GL_LINEAR);//而不是使用NEARST是因为防止锯齿
        GLES20.glTexParameterf(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                GL10.GL_TEXTURE_MAG_FILTER, GL10.GL_LINEAR);
        GLES20.glTexParameterf(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                GL10.GL_TEXTURE_WRAP_S, GL10.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameterf(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
                GL10.GL_TEXTURE_WRAP_T, GL10.GL_CLAMP_TO_EDGE);
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, 0);
        return tex[0];
    }
}