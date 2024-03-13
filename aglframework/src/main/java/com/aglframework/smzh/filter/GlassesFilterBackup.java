package com.aglframework.smzh.filter;

import static android.opengl.GLES20.GL_TRIANGLE_STRIP;

import android.content.Context;
import android.graphics.Bitmap;
import android.opengl.GLES20;
import android.util.Log;

import com.aglframework.smzh.AGLFilter;
import com.aglframework.smzh.OpenGlUtils;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

/**
 * @date 2023/12/14
 * 73为中心点
 * 根据素材大小计算最终的四个顶点，素材大小根据人脸宽度需要更改顶点坐标
 * 优化：需要加入旋转矩阵
 */
public class GlassesFilterBackup extends AGLFilter {
    private String TAG = "uu";
    private FloatBuffer stickerCubeBuffer = ByteBuffer.allocateDirect(8 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
    private FloatBuffer stickerTextureBuffer = ByteBuffer.allocateDirect(8 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
    private Bitmap bitmap;
    private int stickerTexture = -1;

    private boolean enable = false;

    private float[] stickerCube;

    private final float[] stickerCoordnate = {
            0.0f, 1.0f,
            1.0f, 1.0f,
            0.0f, 0.0f,
            1.0f, 0.0f
    };
   /* private final float[] stickerCoordnate = {
           0.077778f, 0.515556f,//1
           0.076667f, 0.633333f,//9
           0.076667f, 0.760000f,//10
           0.187778f, 0.282222f,//43
           0.487778f, 0.775556f,//73
           0.385556f, 0.426667f,//46
           0.584444f, 0.444444f,//97
           0.724444f, 0.393333f,//99
           0.567778f, 0.671111f,//81
           0.887778f, 0.515556f,//17
           0.896667f, 0.896667f,//26
//           0.266667f, 0.382222f,//45
//             0.835556f, 0.395556f,//101
//             0.897778f, 0.633333f,//25
//           0.418889f, 0.677778f,//75
   };*/

    FaceDetector faceDetector;
    public GlassesFilterBackup(Context context){
        super(context);
    }

    @Override
    protected void onInit() {
    }

    @Override
    protected void onDrawArraysAfter(Frame frame) {
        if (!enable){
            return;
        }

        if(null != faceDetector){
            stickerCube = faceDetector.onFaceDetected(frame);
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

        GLES20.glEnable(GLES20.GL_BLEND);
        //两个混合方法都可以用
//        GLES20.glBlendFuncSeparate(GLES20.GL_ONE, GLES20.GL_ONE_MINUS_SRC_ALPHA, GLES20.GL_ZERO, GLES20.GL_ONE);
//        GLES20.glBlendFunc(GLES20.GL_ONE, GLES20.GL_ONE_MINUS_SRC_ALPHA);
        GLES20.glBlendFunc(GLES20.GL_SRC_ALPHA, GLES20.GL_ONE_MINUS_SRC_ALPHA);

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

    public interface FaceDetector{
        public float[] onFaceDetected(Frame frame);
    }
    public void setFaceDetector(FaceDetector faceDetector){
        this.faceDetector = faceDetector;
    }

    public void switchAvailable(){
        enable = !enable;
    }

    public boolean getAvailable(){
        return enable;
    }

}
