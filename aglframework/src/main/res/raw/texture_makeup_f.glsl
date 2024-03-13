varying highp vec2 textureCoordinate;

uniform sampler2D inputImageTexture;
uniform mediump float strength;             // 彩妆强度
uniform int makeupType;             // 彩妆类型, 0表示原图，1比表示绘制没有遮罩的素材，2主要表示美瞳裁剪，3表示绘制唇彩
void main()
{
    if(makeupType == 0){
        gl_FragColor = texture2D(inputImageTexture, textureCoordinate.xy);
    }else if(makeupType == 1){
        lowp vec4 textureColor = texture2D(inputImageTexture, textureCoordinate.xy);
        gl_FragColor = textureColor * strength;
    }else{
        gl_FragColor = texture2D(inputImageTexture, textureCoordinate.xy);
    }

}