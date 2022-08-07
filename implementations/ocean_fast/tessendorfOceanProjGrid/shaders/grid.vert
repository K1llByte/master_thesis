


uniform mat4 m_pi, m_vi;
uniform mat4 m_pvm, m_proj, m_view;
uniform float near, far;
uniform vec3 camPos;

uniform float yLimit = 0.0;
uniform int rows, columns;

uniform sampler2DArray htk;
uniform mat3 m_normal;

uniform int width;
uniform int L;
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


//#define TEXTURE_GRADS

#define i gl_InstanceID

void main() {
	
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
	
	vec4 p,slope;
	vec2 tc = pos.xz / L;
	
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

	vec4 layer0 = textureGrad(htk, vec3(tc, 0), dx/L, dy/L);
	vec4 layer1 = textureGrad(htk, vec3(tc, 1), dx/L, dy/L);

	p.xz =  pos.xz - layer1.xy * choppyFactor;
	p.y = layer0.x;

//#define JACOBIAN
/*
#ifdef JACOBIAN
	vec2 dDdx = (textureGradOffset(dxz, tc, dx/L, dy/L, ivec2(1,0)).xz -
			     textureGradOffset(dxz, tc, dx/L, dy/L, ivec2(-1,0)).xz ) ;
	vec2 dDdy = (textureGradOffset(dxz, tc, dx/L, dy/L, ivec2(0,1)).xz -
			     textureGradOffset(dxz, tc, dx/L, dy/L, ivec2(0,-1)).xz ) ;
	DataOut.jacobian = (1 + dDdx.x) * (1 + dDdy.y) * dDdx.y * dDdy.x;
	DataOut.jacxy = vec2(dDdx.x, dDdy.y);
#endif
	*/
#else

	vec4 layer0 = texture(htk, vec3(tc, LAYER_Y_JXY_JXX_JYY));
	vec4 layer1 = texture(htk, vec3(tc, LAYER_DX_DZ_SX_SZ));
	p.xz =  pos.xz - layer1.xz * choppyFactor;
	p.y = layer0.x;
	
#endif	

	
	
	
	if (dot(dir, vec3(0,-1,0)) < 0)
		DataOut.flag = 1;
	else 
		DataOut.flag = 0;
	
	DataOut.texCoord = tc;
	
	slope.xz = layer1.zw;
	
	
	// texture coordinates after vertex displacement
	//vec3 normal = normalize(vec3( -slope.x, 1,  -slope.z));
	
	
//	tc = tc + vec2(0.23,0.67);
//	p.xz +=  (pos.xz + texture(dxz, tc).xz * ( 1+choppyFactor/(1+(exp(-windSpeed+20)/5))));
//	p.y +=  (texture(heightMap, tc).r);
//	slope += texture(slope_xz, tc);
	
	vec3 normal = normalize(vec3( -slope.x, 1,  -slope.z));
	
	p.w = 1;
	DataOut.world_norm = normal;
	
	DataOut.pos = p.xyz;
	
	
	
	gl_Position = m_pvm * p;
	
}