precision mediump float;
varying vec2 textureCoordinate;
uniform sampler2D inputImageTexture;

// ͼ��ѿ�������ϵ�Ĺؼ��㣬Ҳ��������������Կ�ߵõ�
uniform vec2 cartesianPoints[106];

// �����̶�
uniform float strength;

// ���������߱��ڼ�������
uniform int textureWidth;
uniform int textureHeight;

// �����α䴦�� -> ��ɲ� http://www.gson.org/thesis/warping-thesis.pdf
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
    float faceLiftScale = strength * 0.05;  // ���ű���
    float radius = faceLength;

    // ��Ӧ�ƶ� FIXME ��������
    currentPoint = cartesianPoints[16];
    destPoint = currentPoint + (cartesianPoints[71] - currentPoint) * faceLiftScale;    // ����Ŀ��� -> �� FIXME ���촽�ƶ�
    coordinate = curveWarp(coordinate, currentPoint, destPoint, radius);                // ���»�������

    currentPoint = cartesianPoints[32];
    destPoint = currentPoint + (cartesianPoints[71] - currentPoint) * faceLiftScale;
    coordinate = curveWarp(coordinate, currentPoint, destPoint, radius);

    // ��Ӧ�ƶ� FIXME �°�����
    radius = faceLength * 0.8;
    currentPoint = cartesianPoints[3];
    destPoint = currentPoint + (cartesianPoints[86] - currentPoint) * (faceLiftScale * 0.6);    // ����Ŀ��� -> �� FIXME �Ǽ��ƶ�
    coordinate = curveWarp(coordinate, currentPoint, destPoint, radius);

    currentPoint = cartesianPoints[19];
    destPoint = currentPoint + (cartesianPoints[86] - currentPoint) * (faceLiftScale * 0.6);
    coordinate = curveWarp(coordinate, currentPoint, destPoint, radius);

    return coordinate;
}

void main() {
    vec2 coordinate = textureCoordinate.xy;

    // ��ǿ��Ϊ0��ֱ�ӻ���
    if (strength == 0.0){
        gl_FragColor = texture2D(inputImageTexture, coordinate);
        return;
    }

    // ������ת��ͼ���С��������Ϊ�˷������
    coordinate = textureCoordinate * vec2(float(textureWidth), float(textureHeight));

    float eyeDistance = distance(cartesianPoints[74], cartesianPoints[77]); // ����ͫ�׵ľ���

    // ����
    coordinate = faceLift(coordinate, eyeDistance);

    // ת���ԭ������������ϵ
    coordinate = coordinate / vec2(float(textureWidth), float(textureHeight));

    // ���ͼ��
    gl_FragColor = texture2D(inputImageTexture, coordinate);
}