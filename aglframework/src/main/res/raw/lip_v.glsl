attribute vec4 position;                // 图像顶点坐标
attribute vec4 inputTextureCoordinate;  // 图像纹理坐标

varying vec2 textureCoordinate;         // 图像纹理坐标
varying vec2 maskCoordinate;            // 遮罩纹理坐标(new)

void main() {
    gl_Position = position;
    // LUT 纹理坐标，用顶点来计算
    textureCoordinate = position.xy * 0.5 + 0.5;
    // 遮罩纹理坐标，用传进来的坐标值计算
    maskCoordinate = inputTextureCoordinate.xy;
}