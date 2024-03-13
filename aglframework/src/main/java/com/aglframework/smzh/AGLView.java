package com.aglframework.smzh;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.PixelFormat;
import android.graphics.SurfaceTexture;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.os.Handler;
import android.os.Looper;
import android.util.AttributeSet;
import android.util.Log;

import com.aglframework.smzh.filter.RenderScreenFilter;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.LinkedList;
import java.util.Queue;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

public class AGLView extends /*GLView*/GLSurfaceView {

    private AGLRenderer renderer;
    private volatile boolean needCapture = false;

    public AGLView(Context context) {
        super(context);
        init();
    }

    public AGLView(Context context, AttributeSet attributeSet) {
        super(context, attributeSet);
        init();
    }

    public void init() {
        if (renderer == null) {
            renderer = new AGLRenderer();
            setEGLContextClientVersion(2);
            setEGLConfigChooser(8, 8, 8, 8, 16, 0);
            getHolder().setFormat(PixelFormat.RGBA_8888);
            setRenderer(renderer);
            setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);
        }
    }

    public void setRendererSource(ISource rendererSource) {
        renderer.setSource(rendererSource);
        requestRender();
    }

    public void clear() {
        renderer.clear();
        requestRender();
    }

    public void setFilter(final IFilter filter) {
        renderer.setFilter(filter);
        requestRender();
    }

    public void setDisabled(boolean disable) {
        renderer.setDisable(disable);
        requestRender();
    }

    public int getImageWidth() {
        if (renderer.getSource() != null) {
            return renderer.getSource().getWidth();
        } else {
            return 0;
        }
    }

    public int getImageHeight() {
        if (renderer.getSource() != null) {
            return renderer.getSource().getHeight();
        } else {
            return 0;
        }
    }

    public void capture(CaptureListener captureListener) {
        if (!needCapture) {
            needCapture = true;
            renderer.captureListener = captureListener;
        }
    }

    private class AGLRenderer implements GLSurfaceView.Renderer {

        private ISource iSource;
        private RenderScreenFilter screenFilter;
        private final Queue<Runnable> runOnDraw;
        private final Queue<Runnable> runOnDrawEnd;
        private IFilter filter;
        private boolean disable;
        private int outWidth;
        private int outHeight;
        private CaptureListener captureListener;
        private volatile boolean isCapturing = false;

        private int count = 0;

        AGLRenderer() {
            screenFilter = new RenderScreenFilter(getContext());
            runOnDraw = new LinkedList<>();
            runOnDrawEnd = new LinkedList<>();

            count = 0;
        }

        @Override
        public void onSurfaceCreated(GL10 gl, EGLConfig config) {
            screenFilter.setTextureCoordination(Transform.TEXTURE_ROTATED_180);
        }

        @Override
        public void onSurfaceChanged(GL10 gl, int width, int height) {
            this.outWidth = width;
            this.outHeight = height;
            screenFilter.setOutSize(width, height);
        }


        @Override
        public void onDrawFrame(GL10 gl) {
            GLES20.glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);
//            if (count % 60 == 0) {
//                Log.d("onDrawFrame", "start rendering" + System.currentTimeMillis());
//            }

            runAll(runOnDraw);

            if (iSource != null) {
                IFilter.Frame frame = iSource.createFrame();

//                Log.d("uu", "onDrawFrame: bid1 "+frame.getFrameBuffer());
                if (filter != null && !disable) {
                    frame = filter.draw(frame);
                }

//                Log.d("uu", "onDrawFrame: bid2 " + frame.getFrameBuffer());
                if (needCapture && !isCapturing) {
                    isCapturing = true;
                    capture(frame);
                }
//                Log.d("uu", "onDrawFrame: bid3 "+frame.getFrameBuffer());

                screenFilter.setVerticesCoordination(Transform.adjustVetices(frame.getTextureWidth(), frame.getTextureHeight(), outWidth, outHeight));
                screenFilter.draw(frame);
            }

            runAll(runOnDrawEnd);


//            if (count % 60 == 0) {
//                Log.d("onDrawFrame", "rendering finished" + System.currentTimeMillis());
//            }

//            count++;
        }

        private void capture(IFilter.Frame frame) {
//            Log.d("uu", "onDrawFrame: bid2.2 " + frame.getFrameBuffer());
            ByteBuffer captureBuffer = ByteBuffer.allocate(getImageWidth() * getImageHeight() * 4);
            captureBuffer.order(ByteOrder.nativeOrder());
            captureBuffer.rewind();
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, frame.getFrameBuffer());
            GLES20.glReadPixels(0, 0, frame.getTextureWidth(), frame.getTextureHeight(), GLES20.GL_RGBA, GL10.GL_UNSIGNED_BYTE, captureBuffer);
            final Bitmap bmp = Bitmap.createBitmap(frame.getTextureWidth(), frame.getTextureHeight(), Bitmap.Config.ARGB_8888);
            bmp.copyPixelsFromBuffer(captureBuffer);

            Matrix matrix = new android.graphics.Matrix();
            matrix.postScale(-1, 1);
            final Bitmap result = Bitmap.createBitmap(bmp, 0, 0, bmp.getWidth(), bmp.getHeight(), matrix, true);

            needCapture = false;
            new Handler(Looper.getMainLooper()).post(new Runnable() {
                @Override
                public void run() {
                    if (captureListener != null) {
                        captureListener.captured(result);
                        isCapturing = false;
                    }
                }
            });
        }

        void setSource(ISource iSource) {
            this.iSource = iSource;
        }

        void clear() {
            if (iSource != null) {
                iSource.destroy();
                iSource = null;
            }

            if (screenFilter != null) {
                screenFilter.destroy();
            }

        }

        void setFilter(final IFilter filter) {
            runOnDraw(new Runnable() {
                @Override
                public void run() {
                    final IFilter oldFilter = AGLRenderer.this.filter;
                    AGLRenderer.this.filter = filter;
                    if (oldFilter != null) {
                        oldFilter.destroy();
                    }
                }
            });
        }

        private void runAll(Queue<Runnable> queue) {
            synchronized (runOnDraw) {
                while (!queue.isEmpty()) {
                    queue.poll().run();
                }
            }
        }

        private void runOnDraw(final Runnable runnable) {
            synchronized (runOnDraw) {
                runOnDraw.add(runnable);
            }
        }

        protected void runOnDrawEnd(final Runnable runnable) {
            synchronized (runOnDrawEnd) {
                runOnDrawEnd.add(runnable);
            }
        }

        void setDisable(boolean disable) {
            this.disable = disable;
        }

        public ISource getSource() {
            return iSource;
        }
    }

    public interface CaptureListener {
        void captured(Bitmap bitmap);
    }

}
