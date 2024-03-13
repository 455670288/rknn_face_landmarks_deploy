package com.rockchip.gpadc.demo.view;

import android.content.Context;
import android.opengl.GLSurfaceView;
import android.util.AttributeSet;

import com.rockchip.gpadc.demo.renderer.XYRenderQuadrilateral;

/**
 * @author liuml
 * @explain
 * @time 2018/11/15 11:17
 */
public class XYGLSurfaceView extends GLSurfaceView {
    public XYGLSurfaceView(Context context) {
        this(context,null);
    }

    public XYGLSurfaceView(Context context, AttributeSet attrs) {
        super(context, attrs);
        //设置OpenGL版本
        setEGLContextClientVersion(2);
//        setRenderer(new XYRenderImage(getContext()));
        setRenderer(new XYRenderQuadrilateral(getContext()));
//        setRenderer(new XYRenderTriangle(getContext()));

    }
}
