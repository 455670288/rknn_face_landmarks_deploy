precision mediump float;

varying vec2 textureCoordinate;     // 图像纹理坐标

uniform sampler2D inputImageTexture;  // 相机纹理，不用传，自带

uniform sampler2D inputTexture;     // LUT纹理

varying vec2 maskCoordinate;        // 遮罩纹理坐标

uniform sampler2D maskTexture;      // 遮罩纹理

void main() {
    // 获取相机颜色
    lowp vec4 textureColor = texture2D(inputImageTexture, textureCoordinate.xy);

    // 获取遮罩颜色
    lowp vec4 lipMaskColor = texture2D(maskTexture, maskCoordinate.xy);

    // 遮罩中非黑色地区：修改颜色
    if (lipMaskColor.r > 0.005) {
        mediump vec2 quad1;
        mediump vec2 quad2;
        mediump vec2 texPos1;
        mediump vec2 texPos2;

        mediump float blueColor = textureColor.b * 15.0;    // LUT维度数-1

        quad1.y = floor(floor(blueColor) / 4.0);            // sqrt(维度数)
        quad1.x = floor(blueColor) - (quad1.y * 4.0);

        quad2.y = floor(ceil(blueColor) / 4.0);
        quad2.x = ceil(blueColor) - (quad2.y * 4.0);

        texPos1.xy = (quad1.xy * 0.25) + 0.5/64.0 + ((0.25 - 1.0/64.0) * textureColor.rg);
        texPos2.xy = (quad2.xy * 0.25) + 0.5/64.0 + ((0.25 - 1.0/64.0) * textureColor.rg);

        lowp vec3 newColor1 = texture2D(inputTexture, texPos1).rgb;
        lowp vec3 newColor2 = texture2D(inputTexture, texPos2).rgb;

        lowp vec3 newColor = mix(newColor1, newColor2, fract(blueColor));

        textureColor = vec4(newColor, 1.0) * lipMaskColor.r;
    } else {
        // 否则置为原来的颜色
        textureColor = texture2D(inputImageTexture, textureCoordinate.xy);
    }

    gl_FragColor = textureColor;
}
