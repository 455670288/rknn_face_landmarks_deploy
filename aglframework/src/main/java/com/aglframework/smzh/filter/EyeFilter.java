package com.aglframework.smzh.filter;

import android.content.Context;
import android.graphics.Bitmap;
import android.opengl.GLES20;
import android.util.Log;

import com.aglframework.smzh.AGLFilter;
import com.aglframework.smzh.OpenGlUtils;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;

import static android.opengl.GLES20.GL_ELEMENT_ARRAY_BUFFER;
import static android.opengl.GLES20.GL_TRIANGLES;
import static android.opengl.GLES20.GL_UNSIGNED_SHORT;
import static android.opengl.GLES20.glBindBuffer;
import static android.opengl.GLES20.glBufferData;
import static android.opengl.GLES20.glDrawElements;

/**
 * @date 2023/12/14
 */
public class EyeFilter extends AGLFilter {
    private String TAG = "uu";

    // 坐标数 * 4（浮点）
    private FloatBuffer stickerCubeBuffer = ByteBuffer.allocateDirect(88 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
    private FloatBuffer stickerTextureBuffer = ByteBuffer.allocateDirect(88 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();

    // 顶点数 * 2（整型）
    private ShortBuffer indexBuffer = ByteBuffer.allocateDirect(144 * 2).order(ByteOrder.nativeOrder()).asShortBuffer();
    private Bitmap bitmap;
    private int stickerTexture = -1;


    private float[] stickerCube;

    private final float[] stickerCoordnate = {
            0.0f, 1.0f,
            1.0f, 1.0f,
            0.0f, 0.0f,
            1.0f, 0.0f
    };

    // 注意画三角的顺序要一致
    // 一共48个三角形144个顶点 -> indexBuffer
    private final short[] indices = {
            0, 1, 22,
            22, 1, 21,
            21, 1, 2,
            21, 2, 3,
            21, 3, 42,
            42, 3, 4,
            42, 4, 5,
            42, 5, 6,
            42, 6, 11,
            11, 6, 7,
            11, 7, 8,
            11, 8, 10,
            10, 8, 9,   // 13

            12, 42, 11,
            13, 12, 11,
            14, 12, 13,
            35, 14, 13,
            35, 13, 34,
            43, 35, 34,
            43, 34, 29,
            29, 34, 30,
            30, 34, 31,
            31, 34, 33,
            31, 33, 32, // 11

            28, 43, 29,
            27, 43, 28,
            26, 43, 27,
            26, 40, 43,
            25, 40, 26,
            24, 40, 25,
            24, 41, 40,
            23, 41, 24, // 8

            40, 39, 43,
            39, 38, 43,
            38, 35, 43,
            38, 37, 35,
            37, 36, 35,
            36, 15, 35,
            36, 16, 15,
            16, 17, 15,
            15, 17, 12,
            15, 12, 14,
            35, 15, 14, // 11

            17, 18, 12,
            18, 19, 12,
            12, 19, 42,
            19, 20, 42,
            20, 21, 42  // 5

    };

    // 用到44个关键点88个坐标 -> stickerCubeBuffer
    private final float[] stickerCoordinate = {
            0.185547f, 0.343750f, // 9

            0.185547f, 0.410156f, // 10
            0.189453f, 0.464844f, // 11
            0.193359f, 0.511719f, // 12
            0.203125f, 0.564453f, // 13
            0.224609f, 0.613281f, // 14

            0.242188f, 0.658203f, // 15
            0.269531f, 0.705078f, // 16
            0.302734f, 0.740234f, // 2
            0.347656f, 0.761719f, // 3
            0.404297f, 0.742188f, // 52

            0.435547f, 0.646484f, // 77
            0.457031f, 0.593750f, // 76
            0.490234f, 0.580078f, // 86 公用13
            0.490234f, 0.542969f, // 74
            0.490234f, 0.503906f, // 73

            0.490234f, 0.457031f, // 72 公用16
            0.457031f, 0.478516f, // 75
            0.457031f, 0.445313f, // 39
            0.398438f, 0.416016f, // 37
            0.359375f, 0.390625f, // 33

            0.326172f, 0.365234f, // 36
            0.291016f, 0.349609f, // 35
            0.248047f, 0.343750f, // 25
            0.791016f, 0.410156f, // 26
            0.787109f, 0.464844f, // 27

            0.783203f, 0.511719f, // 28
            0.773438f, 0.564453f, // 29
            0.751953f, 0.613281f, // 30
            0.734375f, 0.658203f, // 31
            0.707031f, 0.705078f, // 32

            0.673828f, 0.740234f, // 18
            0.628906f, 0.761719f, // 19
            0.572266f, 0.742188f, // 61
            0.541016f, 0.646484f, // 83
            0.519531f, 0.593750f, // 82

            0.519531f, 0.478516f, // 81
            0.578125f, 0.445313f, // 89
            0.617188f, 0.416016f, // 90
            0.650391f, 0.390625f, // 87
            0.685547f, 0.365234f, // 91

            0.728516f, 0.349609f, // 93
            0.322266f, 0.568359f, // 106    额外42
            0.644531f, 0.568359f  // 107    额外43
    };

    FaceDetector faceDetector;

    public EyeFilter(Context context) {
        super(context);
    }

    @Override
    protected void onDrawArraysAfter(Frame frame) {
        if (null != faceDetector) {
            stickerCube = faceDetector.onFaceDetected();
            if(stickerCube == null) return;
//            Log.d(TAG, "onDrawArraysPre:人脸监测执行完毕，顶点坐标赋值成功");
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

        GLES20.glEnable(GLES20.GL_BLEND);
        //两个混合方法都可以用
//        GLES20.glBlendFuncSeparate(GLES20.GL_ONE, GLES20.GL_ONE_MINUS_SRC_ALPHA, GLES20.GL_ZERO, GLES20.GL_ONE);
        GLES20.glBlendFunc(GLES20.GL_ONE, GLES20.GL_ONE_MINUS_SRC_ALPHA);

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
    }

    public void setBitmap(Bitmap bitmap) {
        this.bitmap = bitmap;
    }

    @Override
    public void destroy() {
        super.destroy();
        if (stickerTexture != -1) {
            GLES20.glDeleteTextures(1, new int[]{stickerTexture}, 0);
            stickerTexture = -1;
        }
    }

    public void setFaceDetector(FaceDetector faceDetector) {
        this.faceDetector = faceDetector;
    }

}
