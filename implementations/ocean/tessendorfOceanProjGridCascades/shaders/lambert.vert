

uniform sampler2DArray htk;
uniform vec4 ssss;
uniform float choppyFactor, windSpeed;
uniform int cascadeCount = 4;
uniform vec4 L;

uniform mat4 m_pvm;
uniform mat3 m_normal;

in vec4 position;
in vec3 normal;

out vec3 normalV;

void main() {

	vec2 disp = vec2(50.0,50.0);
	vec4 pos = position;
//	vec2 tc = disp/L[0];// + vec2(0.5, 0.5);
//	pos.xz += disp;// + texture(dxz, tc).xz * choppyFactor;
//	pos.y += texture(htk, vec3(tc, LAYER_DY)).r;

	float height=0;
	vec2 dxz = vec2(0);
	for (int casc = 0; casc < cascadeCount; ++casc) {
		height	+= texture(htk, vec3(vec2(0,0) / L[casc], LAYER_Y))[casc];
		dxz += vec2(texture(htk, vec3(vec2(0,0) / L[casc], LAYER_DX))[casc],
		            texture(htk, vec3(vec2(0,0) / L[casc], LAYER_DZ))[casc]);
	}
	vec4 p;
	p.y = height + pos.y;
	p.xz =  pos.xz - dxz * choppyFactor;
	p.w = 1;
	normalV = normalize(m_normal * normal);
	
	gl_Position = m_pvm * p;
}

