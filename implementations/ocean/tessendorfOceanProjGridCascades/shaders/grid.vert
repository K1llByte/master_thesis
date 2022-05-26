

layout(std430, binding = 0) buffer oceanInfo{
	vec4 info[];
};

uniform mat4 m_pi, m_vi;
uniform mat4 m_pvm, m_proj, m_view;
uniform float near, far;
uniform vec3 camPos;

uniform float yLimit = 0.0;
uniform int rows, columns;

uniform sampler2DArray htk;
uniform int cascadeCount = 4;
uniform int layerCount = 8;
uniform mat3 m_normal;

uniform int width;
uniform vec4 L;
uniform float windSpeed;
uniform float choppyFactor;

uniform float A = 1, Q = 0.3, w = 0.4, phi = 1;
uniform vec2 D = vec2(1,1);
uniform float timer;
uniform vec4 l_dir;


out Data {
	vec3 l_dir;
	vec3 pos;
	vec3 world_norm;
	vec2 texCoord;
	float flag;	
	float jacobian;
	vec2 jacxy;
} DataOut;

in vec4 position;


#define i gl_InstanceID


void main() {
	

	
	vec4 p;
	mat4 mi = m_vi * m_pi;
	
	int col = i / rows;
	int row = i % rows;
	
	vec2 disp;
	disp.x = 1.0 / columns;
	disp.y = 1.0 / rows;
	
	float x = (col*1.0/columns + disp.x*position.x) * 6 - 3.0;
	float y = ((row*1.0/rows + disp.y*position.z) * 6 - 3.0);// * rows * 1.0/columns;
	vec4 pos = vec4(x,y,-near,1);
	pos = mi * pos;
	pos /= pos.w;
	vec3 dir = normalize(pos.xyz-camPos);
	float k = pos.y / dot(dir, vec3(0,-1,0));
	pos.xyz = pos.xyz + k * dir;
	
	vec4 slope;
	float height = 0;
	vec2 dxz = vec2(0);
	
#ifdef TEXTURE_GRADS

	vec4 posXX = vec4(x+disp.x, y, -near, 1);
	posXX = mi * posXX;
	posXX /= posXX.w;
	dir = normalize(posXX.xyz-camPos);
	k = posXX.y / dot(dir, vec3(0,-1,0));
	posXX.xyz = posXX.xyz + k * dir;
	
	vec4 posYY = vec4(x, y+disp.y, -near, 1);
	posYY = mi * posYY;
	posYY /= posYY.w;
	dir = normalize(posYY.xyz-camPos);
	k = posYY.y / dot(dir, vec3(0,-1,0));
	posYY.xyz = posYY.xyz + k * dir;
	
	vec2 dx = (posXX - pos).xz;
	vec2 dy = (posYY - pos).xz;
	
	for (int casc = 0; casc < cascadeCount; ++casc) {
		height	+= textureGrad(htk, vec3(pos.xz / L[casc], LAYER_Y), dx/L[casc], dy/L[casc])[casc];
		dxz += vec2(textureGrad(htk, vec3(pos.xz / L[casc], LAYER_DX), dx/L[casc], dy/L[casc])[casc],
		            textureGrad(htk, vec3(pos.xz / L[casc], LAYER_DZ), dx/L[casc], dy/L[casc])[casc]);
	}
	p.y = height;
	p.xz =  pos.xz - dxz * choppyFactor;

/*	
#if (FOAM == USE_JACOBIAN)
	vec2 dDdx = (textureGradOffset(htk, vec3(pos.xz / L[0], LAYER_DX), dx/L[0], dy/L[0], ivec2(1,0)).xz -
			     textureGradOffset(htk, vec3(pos.xz / L[0], LAYER_DX), dx/L[0], dy/L[0], ivec2(-1,0)).xz ) ;
	vec2 dDdy = (textureGradOffset(htk, vec3(pos.xz / L[0], LAYER_DZ), dx/L[0], dy/L[0], ivec2(0,1)).xz -
			     textureGradOffset(htk, vec3(pos.xz / L[0], LAYER_DZ), dx/L[0], dy/L[0], ivec2(0,-1)).xz ) ;
	DataOut.jacobian = (1 + dDdx.x) * (1 + dDdy.y) * dDdx.y * dDdy.x;
	DataOut.jacxy = vec2(dDdx.x, dDdy.y);
#endif
*/	
#else

	for (int casc = 0; casc < cascadeCount; ++casc) {
		height	+= texture(htk, vec3(pos.xz / L[casc], LAYER_Y))[casc];
		dxz += vec2(texture(htk, vec3(pos.xz / L[casc], LAYER_DX))[casc],
		            texture(htk, vec3(pos.xz / L[casc], LAYER_DZ))[casc]);
	}
	p.xz =  pos.xz - dxz * choppyFactor;
	p.y = height;
		
	
#endif
	p.w = 1;
	
	if (dot(dir, vec3(0,-1,0)) < 0)
		DataOut.flag = 1;
	else 
		DataOut.flag = 0;
	
	DataOut.texCoord = pos.xz;
	DataOut.pos = pos.xyz;
	
	gl_Position = m_pvm * p;
	
//	info[1] = vec4(height, dxz, 0);

	
}