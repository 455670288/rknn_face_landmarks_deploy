package com.aglframework.smzh.filter;

import static android.opengl.GLES20.GL_TRIANGLE_STRIP;

import android.content.Context;
import android.graphics.Bitmap;
import android.opengl.GLES20;
import android.util.Log;

import com.aglframework.smzh.AGLFilter;
import com.aglframework.smzh.OpenGlUtils;
import com.aglframework.smzh.aglframework.R;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

/**
 * @date 2023/12/14
 * 73为中心点//这个是做到一半的Pupil.为了防止改不回去，做一个备份
 * 根据素材大小计算最终的四个顶点，素材大小512*256
 * 直径；L41 - 37；R95 - 91
 * 中心点；L38 R88
 */
public class PupilFilterBackup extends AGLFilter {
    private String TAG = "uu";
    private FloatBuffer stickerCubeBuffer = ByteBuffer.allocateDirect(8 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
    private FloatBuffer stickerTextureBuffer = ByteBuffer.allocateDirect(8 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
    private Bitmap bitmap;
    private int stickerTexture = -1;
    private float strength = 1.0f;

    private float[] stickerCube;
    private float[] stickerCube16;

    private final float[] stickerCoordnate = {
            0.0f, 1.0f,
            1.0f, 1.0f,
            0.0f, 0.0f,
            1.0f, 0.0f
    };
    FaceDetector faceDetector;
    public PupilFilterBackup(Context context){
        super(context, R.raw.texture_makeup_f);
    }
    public PupilFilterBackup(Context context, int fragmentResId) {
        super(context, fragmentResId);
    }
    private int glUniformStrength;//强度
    private int glUniformMakeup;
    @Override
    protected void onInit() {
        super.onInit();
        glUniformStrength = GLES20.glGetUniformLocation(programId, "strength");
        glUniformMakeup = GLES20.glGetUniformLocation(programId,"makeupType");
    }

    @Override
    protected void onDrawArraysAfter(Frame frame) {
        if(null != faceDetector){
            stickerCube16 = faceDetector.onFaceDetected(frame);
            //首先画一只眼睛的瞳孔
            stickerCube = new float[8];
            for(int i = 0;i < 8;i++){
                stickerCube[i] = stickerCube16[i];
            }
            //坐标原点转换为opengl坐标
            for (int i = 0; i < stickerCube.length; i += 2) {
                stickerCube[i] = stickerCube[i] * 2f / frame.getTextureWidth() - 1f;
                stickerCube[i + 1] = stickerCube[i + 1] * 2f / frame.getTextureHeight() - 1f;
            }
        }
        if (stickerTexture == -1 && bitmap != null) {
            stickerTexture = OpenGlUtils.loadTexture(bitmap, OpenGlUtils.NO_TEXTURE, false);
        }

        if (stickerTexture == -1) {
            return;
        }

        if(stickerCube == null){
            Log.d(TAG, "onDrawArraysAfter: 没有人脸监测点就赋值默认的");
            //更改这里的顶点坐标
            stickerCube = new float[8];
            int width = frame.getTextureWidth();
            int height = frame.getTextureHeight();

            stickerCube[0] = width / 2f - 200f;
            stickerCube[1] = height / 2f + 200f;
            stickerCube[2] = width / 2f + 200f;
            stickerCube[3] = height / 2f + 200f;
            stickerCube[4] = width / 2f - 200f;
            stickerCube[5] = height / 2f - 200f;
            stickerCube[6] = width / 2f + 200f;
            stickerCube[7] = height / 2f - 200f;

            //坐标原点转换为opengl坐标
            for (int i = 0; i < stickerCube.length; i += 2) {
                stickerCube[i] = stickerCube[i] * 2f / width - 1f;
                stickerCube[i + 1] = stickerCube[i + 1] * 2f / height - 1f;
            }
        }

        GLES20.glUniform1i(glUniformMakeup,1);
        GLES20.glUniform1f(glUniformStrength,strength);

        GLES20.glEnable(GLES20.GL_BLEND);
        //两个混合方法都可以用
        GLES20.glBlendFuncSeparate(GLES20.GL_ONE, GLES20.GL_ONE_MINUS_SRC_ALPHA, GLES20.GL_ZERO, GLES20.GL_ONE);
//        GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA);

        GLES20.glActiveTexture(GLES20.GL_TEXTURE1);
        bindTexture(stickerTexture);
        GLES20.glUniform1i(glUniformTexture, 1);

        stickerCubeBuffer.clear();
        stickerCubeBuffer.put(stickerCube).position(0);
        GLES20.glVertexAttribPointer(glAttrPosition, 2, GLES20.GL_FLOAT, false, 0, stickerCubeBuffer);
        GLES20.glEnableVertexAttribArray(glAttrPosition);

        stickerTextureBuffer.clear();
        stickerTextureBuffer.put(stickerCoordnate).position(0);
        GLES20.glVertexAttribPointer(glAttrTextureCoordinate, 2, GLES20.GL_FLOAT, false, 0, stickerTextureBuffer);
        GLES20.glEnableVertexAttribArray(glAttrTextureCoordinate);

        GLES20.glDrawArrays(GL_TRIANGLE_STRIP, 0, stickerCube.length/2);

        //绘制R
        update(stickerCube,frame);
        stickerCubeBuffer.clear();
        stickerCubeBuffer.put(stickerCube).position(0);
        GLES20.glVertexAttribPointer(glAttrPosition, 2, GLES20.GL_FLOAT, false, 0, stickerCubeBuffer);
        GLES20.glEnableVertexAttribArray(glAttrPosition);
        GLES20.glDrawArrays(GL_TRIANGLE_STRIP, 0, stickerCube.length/2);

        GLES20.glDisableVertexAttribArray(glAttrPosition);
        GLES20.glDisableVertexAttribArray(glAttrTextureCoordinate);

        GLES20.glDisable(GLES20.GL_BLEND);
        GLES20.glUniform1i(glUniformMakeup,0);
    }

    private void update(float[] stickerCube,Frame frame) {
        for(int i = 8;i < 16;i++){
            stickerCube[i-8] = stickerCube16[i];
        }
        //坐标原点转换为opengl坐标
        for (int i = 0; i < stickerCube.length; i += 2) {
            stickerCube[i] = stickerCube[i] * 2f / frame.getTextureWidth() - 1f;
            stickerCube[i + 1] = stickerCube[i + 1] * 2f / frame.getTextureHeight() - 1f;
        }
    }
    public void setBitmap(Bitmap bitmap) {
        this.bitmap = bitmap;
    }
    public void setStrength(float strength){
        this.strength = strength;
    }
    @Override
    public void destroy() {
        super.destroy();
        if (stickerTexture != -1) {
            GLES20.glDeleteTextures(1, new int[]{stickerTexture}, 0);
            stickerTexture = -1;
        }
    }

    public interface FaceDetector{
        public float[] onFaceDetected(Frame frame);
    }
    public void setFaceDetector(FaceDetector faceDetector){
        this.faceDetector = faceDetector;
    }

}
