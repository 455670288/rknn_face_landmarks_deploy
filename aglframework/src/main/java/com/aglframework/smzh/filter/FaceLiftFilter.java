package com.aglframework.smzh.filter;

import static android.opengl.GLES20.GL_ELEMENT_ARRAY_BUFFER;
import static android.opengl.GLES20.GL_TRIANGLES;
import static android.opengl.GLES20.GL_UNSIGNED_SHORT;
import static android.opengl.GLES20.glBindBuffer;
import static android.opengl.GLES20.glBufferData;
import static android.opengl.GLES20.glDrawElements;

import android.content.Context;
import android.opengl.GLES20;
import android.opengl.GLES30;

import com.aglframework.smzh.AGLFilter;
import com.aglframework.smzh.aglframework.R;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;

import javax.microedition.khronos.opengles.GL10;

public class FaceLiftFilter extends AGLFilter {

    /**
     * 用于接受Activity传来的人脸数据
     */
    private FaceDetector faceDetector;
    /**
     * 模型传入的人脸坐标 [0, 1]
     */
    private float[] stickerCube;
    /**
     * 笛卡尔坐标系
     */
    private float[] mCartesianVertices = new float[106 * 2];

    private float strength;

    // 顶点数 * 2（整型）
    private ShortBuffer indexBuffer;

    /** 顶点坐标缓冲 [-1, 1] -> mPositionHandle -> 顶点着色器 gl_Position*/
    private FloatBuffer mVertexBuffer;
    /** 纹理坐标缓冲 [0, 1] -> mTextureCoordinateHandle -> 顶点着色器 -> 片元着色器 textureCoordinate */
    private FloatBuffer mTextureBuffer;
    /** 笛卡尔坐标缓冲 [0, 宽/高] ，只存106个点 */
    private FloatBuffer mCartesianBuffer;


    /** 笛卡尔坐标系下的关键点句柄，传人脸 */
    private int mCartesianPointsHandle;

    /** 瘦脸强度句柄 */
    private int mStrengthHandle;

    /** 纹理宽高句柄 */
    private int mTextureWidthHandle, mTextureHeightHandle;

    /** 构造函数，需要传入片元着色器实现瘦脸逻辑 */
    public FaceLiftFilter(Context context) {
        super(context, R.raw.face_lift_f);
    }

    @Override
    protected void onInit() {
        // 初始化，获取句柄
        mCartesianPointsHandle = GLES30.glGetUniformLocation(programId, "cartesianPoints");
        mStrengthHandle = GLES30.glGetUniformLocation(programId, "strength");
        mTextureWidthHandle = GLES30.glGetUniformLocation(programId, "textureWidth");
        mTextureHeightHandle = GLES30.glGetUniformLocation(programId, "textureHeight");

        indexBuffer = ByteBuffer.allocateDirect(222 * 3 * 2).order(ByteOrder.nativeOrder()).asShortBuffer();
        mVertexBuffer = ByteBuffer.allocateDirect(118 * 2 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        mTextureBuffer = ByteBuffer.allocateDirect(118 * 2 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        mCartesianBuffer = ByteBuffer.allocateDirect(106 * 2 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
    }

    @Override
    protected void onDrawArraysPre(Frame frame) {
        GLES20.glDisable(GL10.GL_CULL_FACE);
        GLES20.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        GLES20.glClear(GLES30.GL_COLOR_BUFFER_BIT);

        if (null != faceDetector) {
            // 此处传入的是像素位置
            stickerCube = faceDetector.onFaceDetected();

            // 若人脸数据不存在，直接返回
            if (stickerCube == null) return;

            // 坐标原点转换为opengl坐标 [-1, 1]
            for (int i = 0; i < stickerCube.length; i += 2) {
                stickerCube[i] = stickerCube[i] * 2f / frame.getTextureWidth() - 1f;
                stickerCube[i + 1] = stickerCube[i + 1] * 2f / frame.getTextureHeight() - 1f;
            }
            
            // 补充边框点
            stickerCube[220] = -1.0f;
            stickerCube[221] = 1.0f;

            stickerCube[222] = 0f;
            stickerCube[223] = 1.0f;

            stickerCube[224] = 1.0f;
            stickerCube[225] = 1.0f;

            stickerCube[226] = -1.0f;
            stickerCube[227] = 0f;

            stickerCube[228] = 1.0f;
            stickerCube[229] = 0f;

            stickerCube[230] = -1.0f;
            stickerCube[231] = -1.0f;

            stickerCube[232] = 0f;
            stickerCube[233] = -1.0f;

            stickerCube[234] = 1.0f;
            stickerCube[235] = -1.0f;
        }

        mVertexBuffer.clear();
        mVertexBuffer.put(stickerCube).position(0);
        // TODO 传值2 顶点坐标
        GLES20.glVertexAttribPointer(glAttrPosition, 2, GLES20.GL_FLOAT, false, 0, mVertexBuffer);
        GLES20.glEnableVertexAttribArray(glAttrPosition);

        // 计算纹理坐标 （[-1, 1]映射到[0, 1]）
        float[] texturePoints = new float[stickerCube.length];
        for (int i = 0; i < stickerCube.length; i++) {
            texturePoints[i] = stickerCube[i] * 0.5f + 0.5f;
        }

        mTextureBuffer.clear();
        mTextureBuffer.put(texturePoints).position(0);
        // TODO 传值1 纹理坐标
        GLES20.glVertexAttribPointer(glAttrTextureCoordinate, 2, GLES20.GL_FLOAT, false, 0, mTextureBuffer);
        GLES20.glEnableVertexAttribArray(glAttrTextureCoordinate);

        // 坐标原点转换为opengl坐标
        for (int i = 0; i < 106 * 2; i += 2) {
            mCartesianVertices[i] = texturePoints[i] * frame.getTextureWidth();
            mCartesianVertices[i + 1] = texturePoints[i + 1] * frame.getTextureHeight();
        }

        mCartesianBuffer.clear();
        mCartesianBuffer.put(mCartesianVertices);
        mCartesianBuffer.position(0);

        // TODO 传值3 强度、宽高、笛卡尔坐标
        GLES20.glUniform1f(mStrengthHandle, strength);
        GLES20.glUniform1i(mTextureWidthHandle, frame.getTextureWidth());
        GLES20.glUniform1i(mTextureHeightHandle, frame.getTextureHeight());
        GLES20.glUniform2fv(mCartesianPointsHandle, 106, mCartesianBuffer);
    }

    @Override
    protected void onDrawArrays(Frame frame) {
        // 此处指定 glDrawElements
        indexBuffer.clear();
        indexBuffer.put(FaceImageIndices).position(0);

        // 创建并绑定索引缓冲区
        int[] bufferIds = new int[1];
        GLES20.glGenBuffers(1, bufferIds, 0);
        int bufferId = bufferIds[0];
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bufferId);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indexBuffer.capacity() * 2, indexBuffer, GLES20.GL_STATIC_DRAW);

        glDrawElements(GL_TRIANGLES, indexBuffer.capacity(), GL_UNSIGNED_SHORT, 0);
    }


    protected void onDrawArraysAfter(Frame frame) {
    }

    @Override
    public void destroy() {
        super.destroy();
        if (mVertexBuffer != null) {
            mVertexBuffer.clear();
            mVertexBuffer = null;
        }
        if (mTextureBuffer != null) {
            mTextureBuffer.clear();
            mTextureBuffer = null;
        }
        if (indexBuffer != null){
            indexBuffer.clear();
            indexBuffer = null;
        }
    }

    public void setStrength(float strength){
        this.strength = strength;
    }

    public void setFaceDetector(FaceDetector faceDetector) {
        this.faceDetector = faceDetector;
    }


    /**
     * 人脸三角划分共222个三角形
     * 合计 106 + 4 + 8 = 118 个关键点
     */
    private static final short[] FaceImageIndices = {
            // 脸外围
            // 上左 5
            110,113,1,
            110,1,43,
            110,43,48,
            110,48,49,
            110,49,111,

            // 上中 6
            111,49,51,
            111,51,50,
            111,50,72,
            111,72,102,
            111,102,103,
            111,103,104,

            // 上右 5
            112,111,104,
            112,104,105,
            112,105,101,
            112,101,17,
            112,17,114,

            // 中左 6
            1,113,9,
            9,113,10,
            10,113,11,
            11,113,12,
            12,113,13,
            13,113,14,

            // 中右 6
            7,25,114,
            25,26,114,
            26,27,114,
            27,28,114,
            28,29,114,
            29,30,114,

            // 下左 9
            14,113,115,
            15,14,115,
            16,15,115,
            2,16,115,
            3,2,115,
            4,3,115,
            5,4,115,
            6,5,115,
            6,115,116,

            // 下右 9
            114,30,117,
            30,31,117,
            31,32,117,
            32,18,117,
            18,19,117,
            19,20,117,
            20,21,117,
            21,22,117,
            22,116,117,

            // 下中 6
            6,116,7,
            7,116,8,
            8,116,0,
            0,116,24,
            24,116,23,
            23,116,22,

            // 左眉 10
            48,43,44,
            48,44,45,
            108,48,45,
            49,48,108,
            49,108,51,
            108,45,47,
            51,108,47,
            50,51,47,
            50,47,46,
            50,46,72,

            // 右眉 10
            102,72,97,
            102,97,98,
            103,102,98,
            109,103,98,
            104,103,109,
            109,98,99,
            105,104,109,
            105,109,99,
            105,99,100,
            105,100,101,

            // 左眼框 23
            43,1,35,
            1,9,35,
            9,10,35,
            10,11,35,
            35,11,106,
            35,106,36,
            36,106,33,
            33,106,37,
            37,106,76,
            37,76,39,
            39,76,75,
            75,76,73,
            72,75,73,
            72,46,75,
            46,39,75,
            46,47,39,
            47,42,39,
            47,40,42,
            47,45,40,
            45,41,40,
            45,44,41,
            44,35,41,
            44,43,35,

            // 右眼眶 23
            101,93,17,
            17,93,25,
            25,93,26,
            26,93,27,
            27,93,107,
            93,91,107,
            91,87,107,
            87,90,107,
            90,82,107,
            90,89,82,
            89,81,82,
            81,73,82,
            81,72,73,
            97,72,81,
            97,81,89,
            98,97,89,
            98,89,95,
            98,95,94,
            99,97,94,
            99,94,96,
            100,99,96,
            100,96,93,
            101,100,93,

            // 左眼 8
            41,35,36,
            38,41,36,
            40,41,38,
            38,36,33,
            42,40,38,
            38,33,37,
            42,38,37,
            39,42,37,

            // 右眼 8
            95,89,90,
            88,95,90,
            94,95,88,
            88,90,87,
            96,94,88,
            88,87,91,
            96,88,91,
            93,96,91,

            // U区
            // 左半脸 18
            11,12,106,
            12,13,106,
            13,14,106,
            14,15,106,
            106,15,77,
            77,15,52,
            15,16,52,
            16,2,52,
            2,3,52,
            52,3,55,
            3,4,55,
            4,5,55,
            5,6,55,
            55,6,56,
            6,7,56,
            7,8,56,
            56,8,53,
            8,0,53,
            // 右半脸 18
            53,0,24,
            53,24,59,
            59,24,23,
            59,23,22,
            58,59,22,
            58,22,21,
            58,21,20,
            61,58,20,
            61,20,19,
            61,19,18,
            61,18,32,
            61,32,31,
            31,83,61,
            107,83,31,
            107,31,30,
            107,30,29,
            107,29,28,
            107,28,27,

            // 鼻翼 14
            76,106,77,
            76,77,86,
            74,76,86,
            73,76,74,
            73,74,82,
            82,74,86,
            82,86,83,
            82,83,107,
            86,77,78,
            86,78,79,
            86,79,80,
            86,80,85,
            86,85,84,
            86,84,83,

            // 上巴 12
            78,77,52,
            78,52,64,
            79,78,64,
            79,64,63,
            79,63,80,
            80,63,71,
            80,71,67,
            80,67,85,
            85,67,68,
            85,68,84,
            84,68,61,
            84,61,83,

            // 嘴唇 20
            52,55,65,
            65,55,54,
            54,55,56,
            54,56,60,
            60,56,53,
            60,53,59,
            60,59,57,
            57,59,58,
            57,58,69,
            69,58,61,
            68,69,61,
            68,70,69,
            68,67,70,
            67,62,70,
            71,62,67,
            63,62,71,
            63,66,62,
            64,66,63,
            64,65,66,
            64,52,65,

            // 口 6
            65,54,66,
            66,54,62,
            62,54,60,
            62,60,57,
            62,57,70,
            70,57,69
    };
}
