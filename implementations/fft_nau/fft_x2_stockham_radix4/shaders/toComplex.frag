#version 430

uniform sampler2D texUnit;
uniform sampler2D texUnit2;
uniform sampler2D texUnit3;
uniform sampler2D texUnit4;

uniform int channels = 1;

in vec2 texCoord;

out vec4 complex;

void main() {
	
	vec3 texColor = texture(texUnit, texCoord).rgb;
	vec3 texColor2 = texture(texUnit2, texCoord).rgb;
	vec3 texColor3 = texture(texUnit3, texCoord).rgb;
	vec3 texColor4 = texture(texUnit4, texCoord).rgb;
	
	float luminance;
	float luminance2;
	float luminance3;
	float luminance4;
	// for black and white images (single channel)
	if (channels == 1) {
		luminance = texColor.r;
	}
	else {
		// for color images - converto to luminance with same factors as DevIL 
		// https://github.com/DentonW/DevIL/blob/master/DevIL/src-IL/src/il_convert.cpp
		luminance  = dot(vec3(0.212671, 0.715160, 0.072169), texColor);
		luminance2 = dot(vec3(0.212671, 0.715160, 0.072169), texColor2);
		luminance3 = dot(vec3(0.212671, 0.715160, 0.072169), texColor3);
		luminance4 = dot(vec3(0.212671, 0.715160, 0.072169), texColor4);
	}
		
	// complex = vec4(luminance, luminance);
	complex = vec4(luminance, luminance2, luminance3, luminance4);
}