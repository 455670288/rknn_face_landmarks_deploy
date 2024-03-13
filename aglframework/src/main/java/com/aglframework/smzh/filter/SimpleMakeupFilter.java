package com.aglframework.smzh.filter;

import static android.opengl.GLES20.GL_ELEMENT_ARRAY_BUFFER;
import static android.opengl.GLES20.GL_TRIANGLES;
import static android.opengl.GLES20.GL_UNSIGNED_SHORT;
import static android.opengl.GLES20.glBindBuffer;
import static android.opengl.GLES20.glBufferData;
import static android.opengl.GLES20.glDrawElements;

import android.content.Context;
import android.graphics.Bitmap;
import android.opengl.GLES20;
import android.opengl.GLES30;

import com.aglframework.smzh.AGLFilter;
import com.aglframework.smzh.OpenGlUtils;
import com.aglframework.smzh.aglframework.R;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;

public class SimpleMakeupFilter extends AGLFilter {

    private static final String TAG = SimpleMakeupFilter.class.getSimpleName();

    // 坐标数 * 4（浮点）
    private FloatBuffer stickerCubeBuffer;
    private FloatBuffer stickerTextureBuffer;

    // 顶点数 * 2（整型）
    private ShortBuffer indexBuffer;

    private int coordinateCount = -1;
    private int triangleCount = -1;

    /**
     * 纹理图像资源
     */
    private Bitmap bitmap;

    /**
     * 素材纹理ID
     */
    private int stickerTexture = -1;

    /**
     * 用于接受Activity传来的人脸数据
     */
    private FaceDetector faceDetector;

    /**
     * 模型传入的人脸坐标
     */
    private float[] stickerCube;

    // 注意画三角的顺序要一致
    /**
     * 三角形划分 -> indexBuffer
     */
    private short[] indices;

    /**
     * 关键点 -> stickerCubeBuffer
     */
    private float[] stickerCoordinate;

    /** 强度句柄 */
    private int glUniformStrength;

    /** 绘图类型句柄 */
    private int glUniformMakeup;

    /** 强度 */
    private float strength;

    public SimpleMakeupFilter(Context context) {
        super(context, R.raw.texture_makeup_f);
    }

    @Override
    protected void onInit() {
        glUniformStrength = GLES30.glGetUniformLocation(programId, "strength");
        glUniformMakeup = GLES20.glGetUniformLocation(programId,"makeupType");
    }

    /** 初始化缓冲 */
    private void initBuffer(){
        // 坐标数 * 4（浮点）
        if (coordinateCount != -1) {
            stickerCubeBuffer = ByteBuffer.allocateDirect(coordinateCount * 2 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
            stickerTextureBuffer = ByteBuffer.allocateDirect(coordinateCount * 2 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        }

        // 顶点数 * 2（整型）
        if (triangleCount != -1) {
            indexBuffer = ByteBuffer.allocateDirect(triangleCount * 3 * 2).order(ByteOrder.nativeOrder()).asShortBuffer();
        }
    }

    @Override
    protected void onDrawArraysPre(Frame frame){
    }

    @Override
    protected void onDrawArraysAfter(Frame frame) {

        initBuffer();

        if (null != faceDetector) {
            stickerCube = faceDetector.onFaceDetected();

            // 若人脸数据不存在，直接返回
            if (stickerCube == null) return;

            //坐标原点转换为opengl坐标
            for (int i = 0; i < stickerCube.length; i += 2) {
                stickerCube[i] = stickerCube[i] * 2f / frame.getTextureWidth() - 1f;
                stickerCube[i + 1] = stickerCube[i + 1] * 2f / frame.getTextureHeight() - 1f;
            }
        }

        // 纹理不存在，创建纹理
        if (stickerTexture == -1 && bitmap != null) {
            stickerTexture = OpenGlUtils.loadTexture(bitmap, OpenGlUtils.NO_TEXTURE, false);
        }

        if (stickerTexture == -1) {
            return;
        }

        if (stickerCubeBuffer == null || stickerTextureBuffer == null || indexBuffer == null){
            if (coordinateCount != -1 && triangleCount != -1){
                initBuffer();
            } else {
                return;
            }
        }

        if (indices == null || stickerCoordinate == null){
            return;
        }

        // 指定妆容绘制类型
        GLES20.glUniform1i(glUniformMakeup,1);
        // 将强度赋值给句柄
        GLES20.glUniform1f(glUniformStrength, strength);

        // 启用混合
        GLES20.glEnable(GLES20.GL_BLEND);
        GLES20.glBlendFuncSeparate(GLES20.GL_ONE, GLES20.GL_ONE_MINUS_SRC_ALPHA, GLES20.GL_ZERO, GLES20.GL_ONE);
//        GLES20.glBlendFunc(GLES20.GL_ONE, GLES20.GL_ONE_MINUS_SRC_ALPHA);

        // 启用妆效纹理
        GLES20.glActiveTexture(GLES20.GL_TEXTURE1);
        bindTexture(stickerTexture);
        GLES20.glUniform1i(glUniformTexture, 1);

        stickerCubeBuffer.clear();
        stickerCubeBuffer.put(stickerCube).position(0);
        GLES20.glVertexAttribPointer(glAttrPosition, 2, GLES20.GL_FLOAT, false, 0, stickerCubeBuffer);
        GLES20.glEnableVertexAttribArray(glAttrPosition);

        stickerTextureBuffer.clear();
        stickerTextureBuffer.put(stickerCoordinate).position(0);
        GLES20.glVertexAttribPointer(glAttrTextureCoordinate, 2, GLES20.GL_FLOAT, false, 0, stickerTextureBuffer);
        GLES20.glEnableVertexAttribArray(glAttrTextureCoordinate);

        indexBuffer.clear();
        indexBuffer.put(indices).position(0);
        // 创建并绑定索引缓冲区
        int[] bufferIds = new int[1];
        GLES20.glGenBuffers(1, bufferIds, 0);
        int bufferId = bufferIds[0];
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufferId);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexBuffer.capacity() * 2, indexBuffer, GLES20.GL_STATIC_DRAW);

//        GLES20.glDrawArrays(GL_TRIANGLE_STRIP, 0, stickerCube.length/2);
        glDrawElements(GL_TRIANGLES, indices.length, GL_UNSIGNED_SHORT, 0);

        GLES20.glDisableVertexAttribArray(glAttrPosition);
        GLES20.glDisableVertexAttribArray(glAttrTextureCoordinate);

        GLES20.glDisable(GLES20.GL_BLEND);

        // 恢复默认绘制类型
        GLES20.glUniform1i(glUniformMakeup,0);
    }

    @Override
    public void destroy() {
        super.destroy();
        if (stickerTexture != -1) {
            GLES20.glDeleteTextures(1, new int[]{stickerTexture}, 0);
            stickerTexture = -1;
        }
        if (stickerCubeBuffer != null) {
            stickerCubeBuffer.clear();
            stickerCubeBuffer = null;
        }
        if (stickerTextureBuffer != null) {
            stickerTextureBuffer.clear();
            stickerTextureBuffer = null;
        }
        if (indexBuffer != null){
            indexBuffer.clear();
            indexBuffer = null;
        }
    }

    public void setFaceDetector(FaceDetector faceDetector) {
        this.faceDetector = faceDetector;
    }

    public void setStrength(float strength){
        this.strength = strength;
    }

    public void setBitmap(Bitmap bitmap) {
        this.bitmap = bitmap;
    }

    /** 设定妆效素材 关键点 数量 */
    public void setCoordinateCount(int coordinateCount) {
        this.coordinateCount = coordinateCount;
    }

    /** 设定妆效素材 三角划分 三角数量 */
    public void setTriangleCount(int triangleCount) {
        this.triangleCount = triangleCount;
    }

    /** 设定妆效素材 顶点 坐标 */
    public void setStickerCoordinate(float[] stickerCoordinate) {
        this.stickerCoordinate = stickerCoordinate;
    }

    /** 设定妆效素材 三角划分 顶点顺序 */
    public void setIndices(short[] indices) {
        this.indices = indices;
    }

}
