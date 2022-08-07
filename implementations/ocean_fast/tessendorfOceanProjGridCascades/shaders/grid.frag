

uniform int  cascadeCount;

const float oc[30] = {
		98.2, 95.8, 57.0, // I
		97.5, 95.3, 56.5, // IA
		96.8, 94.7, 56.0, // IB
		94.0, 92.7, 54.0, // II
		89.0, 89.0, 52.0, // III
		84.0, 88.0, 52.0, // 1
		75.0, 82.0, 49.0, // 3
		65.0, 73.0, 45.0, // 5
		49.0, 61.0, 40.0, // 7
		29.0, 46.0, 33.0  // 9
	};


layout(std430, binding = 0) buffer oceanInfo{
	vec4 info[];
};


void main() {

	if (DataIn.flag != 0)
		discard;	
		
	vec2 slope = vec2(0,0);
	for (int casc = 0; casc < cascadeCount; ++casc) {

		slope += vec2(texture(htk, vec3(DataIn.pos.xz/L[casc], LAYER_SX))[casc], 
					  texture(htk, vec3(DataIn.pos.xz/L[casc], LAYER_SZ))[casc]);
	}
	vec3 wn = normalize(vec3( -slope.x,  1,  -slope.y));
//	vec3 wn = normalize(DataIn.normal);
	
	
	vec4 color = computeOceanColor(wn);
	
#if (FOAM != NO_FOAM)
	vec4 foamV = texture(foam, DataIn.texCoord);
	float f = computeFoamFactor();
	outputF = color * (1-f) + foamV * f;
#else
	outputF = color;
#endif		

}


/*

void main() {
	if (DataIn.flag != 0)
		discard;

	vec3 eyeDir = normalize(DataIn.pos - camPos.xyz);

	if (DataIn.flag != 0)
		discard;

	vec2 slope = vec2(0);
	for (int i = 0; i < cascadeCount; ++i) {
		slope += vec2(texture(htk, vec3(DataIn.texCoord/L[i], LAYER_SX))[i], 
		              texture(htk, vec3(DataIn.texCoord/L[i], LAYER_SZ))[i]);
	}
	vec3 wn = normalize(vec3( -slope.x, 1,  -slope.y));
		
	vec2 sunAnglesRad = vec2(sunAngles.x, sunAngles.y) * vec2(M_PI/180);
	vec3 sunDir = vec3(cos(sunAnglesRad.y) * sin(sunAnglesRad.x),
							 sin(sunAnglesRad.y),
							-cos(sunAnglesRad.y) * cos(sunAnglesRad.x));
	
	// compute "diffuse color"
	float intensity = max(0.0, dot(wn, sunDir));

	outputF = vec4(intensity);
	//return;
	

	
	//float k = length(dFdx(wn) + dFdy(wn));

#if (FOAM == USE_VERTICAL_ACCELERATION)
	vec4 aVertical;
	for (int i = 1 ; i < cascadeCount-1; ++i) {
		aVertical[i] = texture(htk, vec3(DataIn.texCoord/L[i], LAYER_VA))[i] * choppyFactor;
	}
	float whiteCap = max(max(aVertical[0], aVertical[1]), max(aVertical[2], aVertical[3]));
//	if (whiteCap > 3) {
//		outputF = vec4(1,0,0,0);
//		return;
//	}
#elif (FOAM == USE_JACOBIAN)
	float jxx= 1, jyy = 1, jxy = 0;
	for (int i = 0 ; i < cascadeCount; ++i) {
		jxx += texture(htk, vec3(DataIn.texCoord/L[i], LAYER_JXX))[i] * choppyFactor;
		jyy += texture(htk, vec3(DataIn.texCoord/L[i], LAYER_JYY))[i] * choppyFactor;
		jxy += texture(htk, vec3(DataIn.texCoord/L[i], LAYER_VA_JXY))[i] * choppyFactor;
	}
	float det = jxx * jyy - jxy*jxy;
	float whiteCap = det;
	
	//slope /= (1 + vec2(jxx,jyy));

#endif

	if (dot(wn, -eyeDir) < 0.0) {
		wn = reflect(wn, -eyeDir); // reflects backfacing normals
	}
/*	
#ifdef USE_NOISE	
	vec2 tc = DataIn.pos.xz *0.1 + timer/10000*windDir;//windSpeed;
	float left = texture(voronoi, tc - vec2(1.0/width, 0)).r;
	float right = texture(voronoi, tc + vec2(1.0/width, 0)).r;
	float top = texture(voronoi, tc + vec2(0, 1.0/width)).r;
	float bottom = texture(voronoi, tc - vec2(0, 1.0/width)).r;	
	vec3 hv = vec3(2.0, (right-left)*2, 0); //hv.y *= 10;
	vec3 vv = vec3(0, (bottom-top)*2, -2.0); //vv.y *= 10;
	vec3 np = normalize(cross(hv,vv));
	vec3 xx = vec3(1,0,0);
	vec3 zz = cross(xx, wn);
	xx = cross(wn, zz);
	
	mat3 tt = mat3(xx,wn,zz);
	wn = normalize(tt * np);
#endif 	
*/
	//outputF = vec4(np*0.5 + 0.5, 1);
	//return;
	//wn = np;
	//k=0;
	//k = length(dFdx(wn) + dFdy(wn));
	//wn = normalize(wn*(1-k) + vec3(0,1,0)*k);
/*	vec3 refl = normalize(reflect (eyeDir, wn));
	if (refl.y < 0)
		refl.y = -refl.y;
		
		
#ifdef COMPUTE_SKY		
	vec4 skyC = vec4(skyColor(vec3(0,1,0), sunDir, vec3(0.0, earthRadius+1, 0.0)),1);
#else
	vec4 skyC = texture(sky, vec2(0.5));
#endif	
		
		
#ifdef COMPUTE_SKY_FOR_REFLECTION		
	vec4 skyR = vec4(skyColor(refl, sunDir, vec3(0.0, earthRadius+1, 0.0)),1);
#else
	float phi = atan(refl.z, refl.x);
	float theta = acos(refl.y);
	float aux = tan(phi);
	float x = sqrt(abs(0.9-cos(theta))/(1+aux*aux));
	float y = aux*x;

	vec2 tcSky = vec2(x, y);
	float k = length(tcSky);
	if (k >= 0.99) 
		tcSky *= 0.99/k;
//	tcSky.x = 1 - tcSky.x;
	tcSky = tcSky * 0.5 + 0.5;
	vec4 skyR = texture(sky, tcSky );
#endif	



	vec3 of_dir = normalize(refract(eyeDir, wn, Eta));
	// bottom of the sea normal
	vec3 of_normal = vec3(0, 1, 0);
	
	vec2 oceanSurfaceHeight = texelFetch(htk, ivec3(0), int(log2(width))).rg;

	// compute efective ocean depth	
	float d = info[0].g;                                                                    
	if (d > oceanSurfaceHeight.g - oceanDepth) {
		d = oceanSurfaceHeight.g - oceanDepth;
		info[0].g = d;
	}
	
	
	vec3 of_point = intersect(DataIn.pos, of_dir, of_normal, d);
	vec3 of_vec = of_point - DataIn.pos;
	//float of_dist = sqrt(pow(of_vec.x, 2) + pow(of_vec.y, 2) + pow(of_vec.z, 2));

#ifdef TESTING_COLOR_DEPTHS
	int interval = 1024/ 10;
	int index = int(floor(gl_FragCoord.x / interval));
	float of_dist = index * 4;
#else
	float of_dist = length(of_vec);
#endif	
	
	float ratio = pow(schlickRatio(eyeDir, wn),0.5);
	
	// compute "diffuse color"
 intensity = max(0.0, dot(wn, sunDir));

#ifdef TESTING_COLORS
	int interval = 1024/ 10;
	int index = int(floor(gl_FragCoord.x / interval));
	vec3 sliceColor = vec3( oc[index*3], oc[index*3]+1, oc[index*3+2]);
	vec4 diff = vec4(sliceColor.bgr*0.01,1) * intensity;
#else
	vec4 diff = vec4(oceanTransmittance.bgr*0.01,1) * intensity;
#endif
	outputF = diff  + 
		// refracted light
		//skyC * vec4(trans * (1-ratio) , 1 ) * 2		+ 
		// reflected
		ratio * (diff * 0.04 //+ skyC * vec4(oceanTransmittance.bgr*0.01,1) *0.3
		// specular
		+ pow(max(0, dot(sunDir, refl)),100) * skyR)
		+ skyR * 0.5
		;
#if (FOAM == USE_VERTICAL_ACCELERATION)

#define minFoam 0
#define maxFoam 7
	vec4 foamV = texture(foam, DataIn.texCoord/L[cascadeCount-1]*2);
	float f = pow(smoothstep(1,7, whiteCap), 2.0);
	f = 2*f;
	//float f = (whiteCap - minFoam)/(maxFoam-minFoam);
	//f = -8 + f * 16;
	//f = 1 / (1+exp(-f));
	outputF = outputF * (1-f) + foamV * f;
	//if (whiteCap > 3)
	//	outputF = vec4(1,0,0,0);
		
#elif (FOAM == USE_JACOBIAN)

	vec4 foamV = texture(foam, DataIn.texCoord/L[cascadeCount-1]*2);
	float f = 1-smoothstep(0.0, 0.7, whiteCap);
	if (whiteCap < 0.0)
		f = 1;
	outputF = outputF * (1-f) + foamV * f;
	//outputF = vec4(1,0,0,0);
//	if (abs(1-whiteCap) < 0.01	)
//	outputF = vec4(1,0,0,0);
	
#endif

	//outputF = vec4(wn, 1.0);
}

*/