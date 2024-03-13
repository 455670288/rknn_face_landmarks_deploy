package com.aglframework.smzh.filter;

import static android.opengl.GLES20.GL_ELEMENT_ARRAY_BUFFER;
import static android.opengl.GLES20.GL_TRIANGLES;
import static android.opengl.GLES20.GL_UNSIGNED_SHORT;
import static android.opengl.GLES20.glBindBuffer;
import static android.opengl.GLES20.glBufferData;
import static android.opengl.GLES20.glDrawElements;

import android.content.Context;
import android.graphics.Bitmap;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.opengl.GLES30;
import android.util.Log;

import com.aglframework.smzh.AGLFilter;
import com.aglframework.smzh.OpenGlUtils;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;

public class LipFilter extends AGLFilter {
    private static final String TAG = AGLFilter.class.getSimpleName();

    private int mMaskTextureHandle;
    private int mInputTextureHandle;

    // 坐标数 * 4（浮点）
    private FloatBuffer stickerCubeBuffer = ByteBuffer.allocateDirect(72 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
    private FloatBuffer stickerTextureBuffer = ByteBuffer.allocateDirect(72 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();

    // 顶点数 * 2（整型）
    private ShortBuffer indexBuffer = ByteBuffer.allocateDirect(162 * 2).order(ByteOrder.nativeOrder()).asShortBuffer();

    /**
     * 纹理图像资源
     */
    private Bitmap bitmap;

    private Bitmap mask;

    /**
     * 素材纹理ID
     */
    private int stickerTexture = -1;

    /**
     * 遮罩纹理ID
     */
    private int maskTexture = -1;

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
     * 一共 54 个三角形 162 个顶点 -> indexBuffer
     */
    private final short[] indices = {
            0, 1, 15,
            15, 1, 16,
            16, 1, 2,
            16, 2, 17,
            17, 2, 3,
            18, 17, 3,
            18, 3, 4,
            19, 18, 4,
            20, 19, 4,
            20, 4, 5,
            21, 20, 5,
            21, 5, 6,
            22, 21, 6,
            7, 22, 6,
            7, 9, 22,
            8, 9, 7,  // 下巴 16

            27, 15, 16,
            14, 15, 27,
            26, 14, 27,
            13, 14, 26,
            25, 13, 26,
            12, 13, 25,
            11, 12, 25,
            24, 11, 25,
            10, 11, 24,
            23, 10, 24,
            9, 10, 23,
            22, 9, 23,    // 人中区域 12

            28, 16, 17,
            29, 28, 17,
            29, 17, 18,
            30, 29, 18,
            30, 18, 19,
            20, 30, 19,
            31, 30, 20,
            21, 31, 20,
            21, 32, 31,
            22, 32, 21,   // 下嘴唇 10

            28, 27, 16,
            35, 27, 28,
            26, 27, 35,
            34, 26, 35,
            25, 26, 34,
            24, 25, 34,
            33, 24, 34,
            23, 24, 33,
            32, 23, 33,
            22, 23, 32,   // 上嘴唇 10

            35, 28, 29,
            34, 35, 29,
            34, 29, 30,
            34, 30, 31,
            33, 34, 31,
            32, 33, 31    // 口部 6

    };

    /**
     * 用到 36 个关键点 72 个坐标 -> stickerCubeBuffer
     */
    private final float[] stickerCoordinate = {
            0.000000f, 0.000000f, // 14

            0.000000f, 0.500000f, // 2
            0.000000f, 1.000000f, // 5
            0.250000f, 1.000000f, // 7
            0.500000f, 1.000000f, // 0
            0.750000f, 1.000000f, // 23

            1.000000f, 1.000000f, // 21
            1.000000f, 0.500000f, // 18
            1.000000f, 0.000000f, // 30
            0.875000f, 0.000000f, // 83
            0.750000f, 0.000000f, // 84

            0.625000f, 0.000000f, // 85
            0.500000f, 0.000000f, // 80
            0.375000f, 0.000000f, // 79
            0.250000f, 0.000000f, // 78
            0.125000f, 0.000000f, // 77

            0.128906f, 0.398438f, // 52
            0.234375f, 0.625000f, // 55
            0.363281f, 0.796875f, // 56
            0.511719f, 0.828125f, // 53
            0.671875f, 0.789063f, // 59

            0.777344f, 0.679688f, // 58
            0.898438f, 0.429688f, // 61
            0.757813f, 0.320313f, // 68
            0.613281f, 0.234375f, // 67
            0.515625f, 0.296875f, // 71

            0.386719f, 0.234375f, // 63
            0.269531f, 0.296875f, // 64
            0.191406f, 0.406250f, // 65
            0.308594f, 0.523438f, // 54
            0.500000f, 0.593750f, // 60

            0.687500f, 0.539063f, // 57
            0.804688f, 0.445313f, // 69
            0.671875f, 0.453125f, // 70
            0.519531f, 0.476563f, // 62
            0.312500f, 0.406250f, // 66
    };

    private final float[] defaultCoordinate = {
            0.0f, 1.0f,
            1.0f, 1.0f,
            0.0f, 0.0f,
            1.0f, 0.0f
    };

    /**
     * 需要指定片元着色器代码
     */
    public LipFilter(Context context, int vertexResId, int fragmentResId) {
        super(context, vertexResId, fragmentResId);
    }

    @Override
    protected void onInit() {
        // FIXME 此处绑定新的句柄
        mMaskTextureHandle = GLES20.glGetUniformLocation(programId, "maskTexture");
        mInputTextureHandle = GLES20.glGetUniformLocation(programId, "inputTexture");
    }

    public void setFaceDetector(FaceDetector faceDetector) {
        this.faceDetector = faceDetector;
    }

    /**
     * 设置唇色LUT
     */
    public void setBitmap(Bitmap bitmap) {
        this.bitmap = bitmap;
    }

    /**
     * 设置嘴唇掩码
     */
    public void setMask(Bitmap bitmap) {
        this.mask = bitmap;
    }

    @Override
    protected void onDrawArraysAfter(Frame frame) {
        if (null != faceDetector) {
            stickerCube = faceDetector.onFaceDetected();

            // 若人脸数据不存在，直接返回
            if (stickerCube == null) return;
//            Log.d(TAG, "onDrawArraysPre:人脸监测执行完毕，顶点坐标赋值成功");

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

        // 遮罩纹理不存在，则创建
        if (maskTexture == -1 && mask != null){
            maskTexture = OpenGlUtils.loadTexture(mask, OpenGlUtils.NO_TEXTURE, false);
        }

        if (stickerTexture == -1 && maskTexture == -1) {
            return;
        }

        GLES20.glEnable(GLES20.GL_BLEND);

        //两个混合方法都可以用
//        GLES20.glBlendFuncSeparate(GLES20.GL_ONE, GLES20.GL_ONE_MINUS_SRC_ALPHA, GLES20.GL_ZERO, GLES20.GL_ONE);
        GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA);

        // 绑定 LUT 纹理
        GLES20.glActiveTexture(GLES20.GL_TEXTURE1);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, stickerTexture);
        GLES20.glUniform1i(mInputTextureHandle, 1);

        // 激活遮罩
        GLES20.glActiveTexture(GLES20.GL_TEXTURE2);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, maskTexture);
        GLES20.glUniform1i(mMaskTextureHandle, 2);

        // 将人脸坐标缓冲传入着色器（模型获取的人脸坐标点）
        stickerCubeBuffer.clear();
        stickerCubeBuffer.put(stickerCube).position(0);
        GLES20.glVertexAttribPointer(glAttrPosition, 2, GLES20.GL_FLOAT, false, 0, stickerCubeBuffer);
        GLES20.glEnableVertexAttribArray(glAttrPosition);

        // 将纹理顶点缓冲传入着色器（手动标记的素材坐标点）
        stickerTextureBuffer.clear();
        stickerTextureBuffer.put(stickerCoordinate).position(0);
        GLES20.glVertexAttribPointer(glAttrTextureCoordinate, 2, GLES20.GL_FLOAT, false, 0, stickerTextureBuffer);
        GLES20.glEnableVertexAttribArray(glAttrTextureCoordinate);

        indexBuffer.clear();
        indexBuffer.put(indices).position(0);

        // 创建并绑定索引缓冲区（用于绘制三角形的顶点顺序）
        int[] bufferIds = new int[1];
        GLES20.glGenBuffers(1, bufferIds, 0);
        int bufferId = bufferIds[0];
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufferId);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexBuffer.capacity() * 2, indexBuffer, GLES20.GL_STATIC_DRAW);

        glDrawElements(GL_TRIANGLES, indices.length, GL_UNSIGNED_SHORT, 0);

        GLES20.glDisableVertexAttribArray(glAttrPosition);
        GLES20.glDisableVertexAttribArray(glAttrTextureCoordinate);

        GLES20.glDisable(GLES20.GL_BLEND);
    }

    @Override
    public void destroy() {
        super.destroy();
        if (stickerTexture != -1) {
            GLES20.glDeleteTextures(1, new int[]{stickerTexture}, 0);
            stickerTexture = -1;
        }
    }
}
