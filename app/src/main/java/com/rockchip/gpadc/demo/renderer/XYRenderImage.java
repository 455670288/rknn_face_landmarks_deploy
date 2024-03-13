package com.rockchip.gpadc.demo.renderer;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.opengl.GLUtils;

import com.rockchip.gpadc.demo.R;
import com.rockchip.gpadc.demo.utils.XYShaderUtil;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

/**
 * @author liuml
 * @explain  渲染图片
 * @time 2018/11/15 11:19
 */
public class XYRenderImage implements GLSurfaceView.Renderer {

    private Context context;
    //顶点坐标
    private final  float[] vertexData={
            -1f,-1f,
            1f,-1f,
            -1f,1f,
            1f,1f
    };

    //纹理坐标  正常
//    private final float[] textureData={
//            0f,1f,
//            1f,1f,
//            0f,0f,
//            1f,0f
//    };
    //纹理坐标 倒立
    private final float[] textureData={
            1f,0f,
            0f,0f,
            1f, 1f,
            0f, 1f
    };


    private FloatBuffer vertexBuffer;//顶点buffer
    private FloatBuffer textureBuffer;//纹理buffer
    private int program;
    private int avPosition;//顶点坐标
    private int afPosition;//纹理坐标
    private int textureId;//纹理id保存
//    private int afColor;


    public XYRenderImage(Context context){
        this.context = context;
        vertexBuffer = ByteBuffer.allocateDirect(vertexData.length * 4)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()
                .put(vertexData);
        vertexBuffer.position(0);


        textureBuffer = ByteBuffer.allocateDirect(textureData.length * 4)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()
                .put(textureData);
        textureBuffer.position(0);
    }

    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        String vertexSource = XYShaderUtil.readRawText(context, R.raw.img_vertex_shader);
        String fragmentSource ;

        fragmentSource = XYShaderUtil.readRawText(context, R.raw.img_fragment_shader);

        program = XYShaderUtil.createProgram(vertexSource, fragmentSource);
        if (program > 0) {
            avPosition = GLES20.glGetAttribLocation(program, "av_Position");
            afPosition = GLES20.glGetAttribLocation(program, "af_Position");

            //创建纹理
            int[] textureIds = new int[1];
            GLES20.glGenTextures(1, textureIds, 0);
            if (textureIds[0] == 0) {
                return;
            }

            textureId = textureIds[0];

            //绑定
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textureId);

            //设置参数 环绕（超出纹理坐标范围）：
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_REPEAT);
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_REPEAT);

            //过滤（纹理像素映射到坐标点
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR);
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR);

//            BitmapFactory.Options options = new BitmapFactory.Options();
//            options.inScaled = false;
            Bitmap bitmap = BitmapFactory.decodeResource(context.getResources(), R.mipmap.test);
            if (bitmap == null) {
                return;
            }
            GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bitmap, 0);
            bitmap.recycle();
            bitmap = null;
        }

    }

    @Override
    public void onSurfaceChanged(GL10 gl, int width,int height) {
        //绘制区域
        GLES20.glViewport(0,0,width,height);
    }

    @Override
    public void onDrawFrame(GL10 gl) {
        //清屏缓冲区
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);
        //利用颜色清屏
        GLES20.glClearColor(1.0f,1.0f,1.0f,1.0f);


        //让program可用
        GLES20.glUseProgram(program);
        //顶点坐标
        GLES20.glEnableVertexAttribArray(avPosition);
        GLES20.glVertexAttribPointer(avPosition, 2, GLES20.GL_FLOAT, false, 8, vertexBuffer);
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
        //纹理坐标
        GLES20.glEnableVertexAttribArray(afPosition);
        GLES20.glVertexAttribPointer(afPosition, 2, GLES20.GL_FLOAT, false, 8, textureBuffer);
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
    }
}
