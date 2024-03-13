precision mediump float;
varying vec2 textureCoordinate;
uniform sampler2D inputImageTexture;

// 图像笛卡尔坐标系的关键点，也就是纹理坐标乘以宽高得到
uniform vec2 cartesianPoints[106];

// 瘦脸程度
uniform float strength;

// 传入纹理宽高便于计算坐标
uniform int textureWidth;
uniform int textureHeight;

// 曲线形变处理 -> 另可参 http://www.gson.org/thesis/warping-thesis.pdf
vec2 curveWarp(vec2 textureCoord, vec2 originPosition, vec2 targetPosition, float radius) {
    vec2 offset = vec2(0.0);
    vec2 result = vec2(0.0);

    vec2 direction = targetPosition - originPosition;

    float infect = distance(textureCoord, originPosition) / radius;

    infect = 1.0 - infect;
    infect = clamp(infect, 0.0, 1.0);
    offset = direction * infect;

    result = textureCoord - offset;

    return result;
}

vec2 faceLift(vec2 currentCoordinate, float faceLength) {
    vec2 coordinate = currentCoordinate;
    vec2 currentPoint = vec2(0.0);
    vec2 destPoint = vec2(0.0);
    float faceLiftScale = strength * 0.05;  // 缩放比例
    float radius = faceLength;

    // 对应移动 FIXME 脸颊两侧
    currentPoint = cartesianPoints[16];
    destPoint = currentPoint + (cartesianPoints[71] - currentPoint) * faceLiftScale;    // 计算目标点 -> 向 FIXME 上嘴唇移动
    coordinate = curveWarp(coordinate, currentPoint, destPoint, radius);                // 更新绘制坐标

    currentPoint = cartesianPoints[32];
    destPoint = currentPoint + (cartesianPoints[71] - currentPoint) * faceLiftScale;
    coordinate = curveWarp(coordinate, currentPoint, destPoint, radius);

    // 对应移动 FIXME 下巴两侧
    radius = faceLength * 0.8;
    currentPoint = cartesianPoints[3];
    destPoint = currentPoint + (cartesianPoints[86] - currentPoint) * (faceLiftScale * 0.6);    // 计算目标点 -> 向 FIXME 鼻尖移动
    coordinate = curveWarp(coordinate, currentPoint, destPoint, radius);

    currentPoint = cartesianPoints[19];
    destPoint = currentPoint + (cartesianPoints[86] - currentPoint) * (faceLiftScale * 0.6);
    coordinate = curveWarp(coordinate, currentPoint, destPoint, radius);

    return coordinate;
}

void main() {
    vec2 coordinate = textureCoordinate.xy;

    // 若强度为0，直接绘制
    if (strength == 0.0){
        gl_FragColor = texture2D(inputImageTexture, coordinate);
        return;
    }

    // 将坐标转成图像大小，这里是为了方便计算
    coordinate = textureCoordinate * vec2(float(textureWidth), float(textureHeight));

    float eyeDistance = distance(cartesianPoints[74], cartesianPoints[77]); // 两个瞳孔的距离

    // 瘦脸
    coordinate = faceLift(coordinate, eyeDistance);

    // 转变回原来的纹理坐标系
    coordinate = coordinate / vec2(float(textureWidth), float(textureHeight));

    // 输出图像
    gl_FragColor = texture2D(inputImageTexture, coordinate);
}