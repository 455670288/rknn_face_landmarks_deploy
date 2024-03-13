package com.aglframework.smzh.camera;

import android.graphics.SurfaceTexture;
import android.hardware.Camera;
import android.util.Log;

import com.aglframework.smzh.AGLView;

import java.io.IOException;
import java.util.List;
import java.util.Objects;

@SuppressWarnings("deprecation")
public class AGLCamera {

    private static final String TAG = AGLCamera.class.getSimpleName();

    private Camera camera;
    private int cameraId;
    private AGLView aglView;
    private int previewWidth;
    private int previewHeight;

    private Object lock;
    private boolean isRenderer = false;


    public AGLCamera(AGLView aglView, int width, int height) {
        this.aglView = aglView;
        this.previewWidth = width;
        this.previewHeight = height;
        if (Camera.getNumberOfCameras() > 1) {
            cameraId = Camera.CameraInfo.CAMERA_FACING_FRONT;
        } else {
            cameraId = Camera.CameraInfo.CAMERA_FACING_BACK;
        }
    }

    public AGLCamera(AGLView aglView) {
        this(aglView, 0, 0);
    }

    @SuppressWarnings("SuspiciousNameCombination")
    public void open() {
        if (camera == null) {
            camera = Camera.open(cameraId);
            Camera.Parameters parameters = camera.getParameters();

            try {
                setCameraParameters();

                if (parameters.getSupportedFocusModes().contains(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO)) {
                    parameters.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO);
                }

                camera.autoFocus(new Camera.AutoFocusCallback() {
                    @Override
                    public void onAutoFocus(boolean success, Camera camera) {

                    }
                });
            } catch (Exception e) {
                e.printStackTrace();
            }

            // 请求渲染
            aglView.setRendererSource(new SourceCamera(aglView.getContext(), this, new SurfaceTexture.OnFrameAvailableListener() {
                @Override
                public void onFrameAvailable(SurfaceTexture surfaceTexture) {
                    if (isRenderer){
//                        callRenderer();
                    }
                }
            }));


        }
    }

    public void close() {
        camera.stopPreview();
        camera.setPreviewCallback(null);
        camera.release();
        camera = null;
        aglView.clear();

    }

    public void enableRenderer(){
        this.isRenderer = true;
    }

    public void setLock(Object lock){
        this.lock = lock;
    }

    private void callRenderer(){
        synchronized (lock){
            Log.e("yyx","Camera 请求渲染:");
            aglView.requestRender();
        }
    }

    public void switchCamera() {
        cameraId = (cameraId + 1) % 2;
        close();
        open();
    }


    public void startPreview(SurfaceTexture surfaceTexture) {
        try {
            camera.setPreviewTexture(surfaceTexture);
            camera.startPreview();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public Camera.Parameters getParameter() {
        return camera.getParameters();
    }


    public int getCameraId() {
        return cameraId;
    }

    private void setCameraParameters() {
        Camera.Parameters parameters;
        boolean checkWH = false;
        parameters = camera.getParameters();
        int nearest_width_index = 0;
        int nearest_width_value = 1920;

        List<Camera.Size> sizes = parameters.getSupportedPreviewSizes();
        for (int i = 0; i < sizes.size(); i++) {
            Camera.Size size = sizes.get(i);

            if (Math.abs(size.width - previewWidth) < nearest_width_value) {
                nearest_width_value = Math.abs(size.width - previewWidth);
                nearest_width_index = i;
            }

            if ((size.width == previewWidth) && (size.height == previewHeight)) {
                checkWH = true;
            }

            Log.v(TAG, "Camera Supported Preview Size = " + size.width + "x" + size.height);
        }
        if (!checkWH) {
            Log.e(TAG, "Camera don't support this preview Size = " + previewWidth + "x" + previewHeight);
            previewWidth = sizes.get(nearest_width_index).width;
            previewHeight = sizes.get(nearest_width_index).height;
        }

        Log.v(TAG, "Use preview Size = " + previewWidth + "x" + previewHeight);

        parameters.setPreviewSize(previewWidth, previewHeight);

        if (parameters.isZoomSupported()) {
            parameters.setZoom(0);
        }

        camera.setParameters(parameters);
        Log.i(TAG, "mCamera0 set parameters success.");
    }

    public void setCallback(Camera.PreviewCallback callback) {
        if (camera != null){
            camera.setPreviewCallback(callback);
        }
    }
}
