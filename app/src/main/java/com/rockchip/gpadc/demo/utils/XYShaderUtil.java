package com.rockchip.gpadc.demo.utils;

import android.content.Context;
import android.opengl.GLES20;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;

/**
 * @author liuml
 * @explain
 * @time 2018/11/15 16:54
 */
public class XYShaderUtil {

    //从raw读取数据
    public static String readRawText(Context context, int rawId) {
        InputStream inputStream = context.getResources().openRawResource(rawId);
        BufferedReader reader = new BufferedReader((new InputStreamReader(inputStream)));
        StringBuffer sb = new StringBuffer();
        String line;
        try {
            while ((line = reader.readLine()) != null) {
                sb.append(line).append("\n");
            }
            reader.close();
        } catch (Exception e) {

        }
        return sb.toString();
    }

    /**
     * 加载shader
     * @param shaderType
     * @param source
     * @return
     */
    public static int loadShader(int shaderType, String source) {

        int shader = GLES20.glCreateShader(shaderType);
        if (shader != 0) {
            GLES20.glShaderSource(shader, source);
            GLES20.glCompileShader(shader);
            int[] compile = new int[1];
            GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, compile, 0);
            if (compile[0] != GLES20.GL_TRUE) {
                LogUtil.e("shader compile error");
                GLES20.glDeleteShader(shader);
                shader = 0;
            }
        }
        return shader;
    }

    public static int createProgram(String vertexSource, String fragmentSource) {

        int vertexShader = loadShader(GLES20.GL_VERTEX_SHADER, vertexSource);
        if (vertexShader == 0) {
            return 0;
        }

        int fragmentShader = loadShader(GLES20.GL_FRAGMENT_SHADER, fragmentSource);
        if (fragmentShader == 0) {
            return 0;
        }
        //创建一个渲染程序
        int program = GLES20.glCreateProgram();
        if (program != 0) {
            //将着色器程序添加到渲染程序中
            GLES20.glAttachShader(program,vertexShader);
            GLES20.glAttachShader(program,fragmentShader);
            //链接源程序
            GLES20.glLinkProgram(program);
            int[] lineSatus = new int[1];
            //检查链接源程序是否成功
            GLES20.glGetProgramiv(program, GLES20.GL_LINK_STATUS, lineSatus, 0);
            if (lineSatus[0] != GLES20.GL_TRUE) {
                LogUtil.e("link program error");
                GLES20.glDeleteProgram(program);
                program = 0;
            }
        }
        return program;
    }







}
