package com.aglframework.smzh;

import android.content.Context;
import android.graphics.Bitmap;
import android.opengl.GLES20;
import android.util.Log;
import android.util.Size;

import com.aglframework.smzh.aglframework.R;

import java.io.FileOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import static android.opengl.GLES20.GL_TRIANGLE_STRIP;

public abstract class AGLFilter implements IFilter {

    protected Context context;

    protected int programId;
    /** 顶点着色器句柄 */
    protected int glAttrPosition;
    /** 片元着色器句柄 */
    protected int glAttrTextureCoordinate;
    protected int glUniformTexture;

    protected Frame frame;
    private int outputWidth;
    private int outputHeight;
    private Size outSize;

    private boolean hasInit;
    private boolean isNeedRendererScreen;

    private FloatBuffer cubeBuffer = ByteBuffer.allocateDirect(8 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
    private FloatBuffer textureBuffer = ByteBuffer.allocateDirect(8 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();

    private int vertexResId;
    private int fragmentResId;

    private float[] cube = Transform.RECTANGLE_VERTICES;
    private float[] textureCords = Transform.RECTANGLE_TEXTURE;

    public AGLFilter(Context context) {
        this(context, R.raw.single_input_v, R.raw.texture_f);
    }

    public AGLFilter(Context context, int fragmentResId) {
        this(context, R.raw.single_input_v, fragmentResId);
    }

    public AGLFilter(Context context, int vertexResId, int fragmentResId) {
        this.context = context;
        this.vertexResId = vertexResId;
        this.fragmentResId = fragmentResId;
    }


    private void init() {
        if (!hasInit) {
            // 加载着色器代码并编译
            programId = OpenGlUtils.loadProgram(context, vertexResId, fragmentResId);
            glAttrPosition = GLES20.glGetAttribLocation(programId, "position");
            glAttrTextureCoordinate = GLES20.glGetAttribLocation(programId, "inputTextureCoordinate");
            glUniformTexture = GLES20.glGetUniformLocation(programId, "inputImageTexture");
            onInit();
            hasInit = true;
        }
    }

    protected void onInit() {
    }

    public Frame draw(Frame frame) {
        init();

        if (programId <= 0) {
            return frame;
        }

        int textureWidth = frame.getTextureWidth();
        int textureHeight = frame.getTextureHeight();
        if (isNeedRendererScreen) {
            updateOutputSize(outSize.getWidth(), outSize.getHeight());
        } else {
            updateOutputSize(textureWidth, textureHeight);
        }
        bindFrameBuffer();

        GLES20.glUseProgram(programId);

        // 绑定纹理坐标缓冲
        cubeBuffer.clear();
        cubeBuffer.put(cube).position(0);
        GLES20.glVertexAttribPointer(glAttrPosition, 2, GLES20.GL_FLOAT, false, 0, cubeBuffer);
        GLES20.glEnableVertexAttribArray(glAttrPosition);

        // 绑定纹理缓冲
        textureBuffer.clear();
        textureBuffer.put(textureCords).position(0);
        GLES20.glVertexAttribPointer(glAttrTextureCoordinate, 2, GLES20.GL_FLOAT, false, 0, textureBuffer);
        GLES20.glEnableVertexAttribArray(glAttrTextureCoordinate);

        onDrawArraysPre(AGLFilter.this.frame);

        // 绑定纹理（相机）
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        bindTexture(frame.getTextureId());
        GLES20.glUniform1i(glUniformTexture, 0);

        onDrawArrays(frame);

        // 解绑
        GLES20.glDisableVertexAttribArray(glAttrPosition);
        GLES20.glDisableVertexAttribArray(glAttrTextureCoordinate);

        bindTexture(0);

        if (!isNeedRendererScreen) {
            onDrawArraysAfter(AGLFilter.this.frame);
            return AGLFilter.this.frame;
        } else {
            return null;
        }
    }

    protected void onDrawArraysPre(Frame frame) {

    }

    protected void onDrawArrays(Frame frame) {
        GLES20.glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    }

    protected void onDrawArraysAfter(Frame frame) {

    }

    protected void bindTexture(int textureId) {
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureId);
    }

    public void setTextureCoordination(float[] coordination) {
        this.textureCords = coordination;
    }

    public void setOutSize(int outputWidth, int outputHeight) {
        outSize = new Size(outputWidth, outputHeight);
    }

    public void setVerticesCoordination(float[] coordination) {
        this.cube = coordination;
    }

    private void updateOutputSize(int width, int height) {
        if (width != outputWidth || height != outputHeight) {
            outputWidth = width;
            outputHeight = height;
            FrameBufferProvider.destroyFrameBuffers(frame);
            frame = FrameBufferProvider.createFrameBuffers(outputWidth, outputHeight);
        }
    }

    private void bindFrameBuffer() {
        if (isNeedRendererScreen) {
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
        } else {
            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, frame.getFrameBuffer());
        }
        GLES20.glViewport(0, 0, outputWidth, outputHeight);
        GLES20.glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);
    }

    public void setNeedRendererOnScreen(boolean isNeedRendererScreen) {
        this.isNeedRendererScreen = isNeedRendererScreen;
    }

    public void destroy() {
        hasInit = false;
        FrameBufferProvider.destroyFrameBuffers(frame);
        frame = null;
        outputWidth = 0;
        outputHeight = 0;
    }
}
