package com.rockchip.gpadc.demo.renderer;

import android.content.Context;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;

import com.rockchip.gpadc.demo.R;
import com.rockchip.gpadc.demo.utils.XYShaderUtil;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

/**
 * @author liuml
 * @explain  四边形
 * @time 2018/11/15 11:19
 */
public class XYRenderQuadrilateral implements GLSurfaceView.Renderer {

    private Context context;
    private final  float[] vertexData={
            -1f,0f,
            0f,-1f,
            0f,1f,
            1f,0f
    };
//    private final  float[] vertexData={
//            -1f,0f,
//            0f,-1f,
//            0f,1f,
//
//            0f,1f,
//            0f,-1f,
//            1f,0f
//    };

    private FloatBuffer vertexBuffer;
    private int program;
    private int avPosition;
    private int afColor;

    public XYRenderQuadrilateral(Context context){
        this.context = context;
        vertexBuffer = ByteBuffer.allocateDirect(vertexData.length * 4)
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer()
                .put(vertexData);
        vertexBuffer.position(0);
    }

    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        String vertexSource = XYShaderUtil.readRawText(context, R.raw.vertex_shader);
        String fragmentSource = XYShaderUtil.readRawText(context, R.raw.fragment_shader);
        program = XYShaderUtil.createProgram(vertexSource, fragmentSource);
        if (program > 0) {
            avPosition = GLES20.glGetAttribLocation(program, "av_Position");
            afColor = GLES20.glGetUniformLocation(program, "af_Color");
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

        //设置颜色 用了float 所以用4 f
        GLES20.glUniform4f(afColor,0f,0f,1f,1f);

        //让program可用
        GLES20.glUseProgram(program);
        GLES20.glEnableVertexAttribArray(avPosition);
        GLES20.glVertexAttribPointer(avPosition, 2, GLES20.GL_FLOAT, false, 8, vertexBuffer);
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
    }
}
