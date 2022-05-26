
uniform mat4 m_pi, m_vi;
uniform mat4 m_pvm, m_proj, m_view;
uniform float near, far;
uniform vec3 camPos;

uniform float yLimit = 0.0;
uniform int rows, columns;

uniform sampler2DArray hkt;
uniform int L;
uniform float choppyFactor, windSpeed;


uniform mat3 m_normal;

in vec4 position;
in vec3 normal;

out vec3 normalV; 

void main() {

	vec2 disp = vec2(50.0,50.0);
	vec4 pos = position;
	vec2 tc = disp/L;// + vec2(0.5, 0.5);
	pos.xz += disp - texture(hkt, vec3(tc,1)).xy * choppyFactor;
	pos.y += texture(hkt, vec3(tc,0)).r;
	
	normalV = normalize(m_normal * normal);
	
	gl_Position = m_pvm * pos;
}

