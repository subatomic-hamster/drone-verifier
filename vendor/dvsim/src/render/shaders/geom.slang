// The renderer's shaders (entry points; renderer.cpp builds one pipeline per use):
//   vsMain/fsMain   geometry: sun + hemisphere light, Blinn-Phong/Fresnel specular, PCF sun shadow, distance haze, sRGB out
//   vsMain/fsLin    the same geometry into the R32F linear-depth target (view-space z in metres, exact)
//   vsShadow        depth only, from the sun
//   vsSky/fsSky     full-screen procedural sky (gradient, warm horizon, sun, noise clouds)
//   vsEdge/fsEdge   CAD-style crease lines: each segment is a screen-space quad of constant pixel width
//   vsRain/fsRain   rain streaks: instanced screen-space quads (instance = one drop in a world-fixed periodic box around the camera,
//                   moving with the drop velocity), alpha-blended, depth-tested against the scene
// The CPU builds everything camera-relative in double (see renderer.cpp); here all values are small-magnitude floats.
struct Scene {
  float4 view_r, view_u, view_f;  // camera basis rows (xyz)
  float4 proj;                    // sx, sy, A, B: clip = (sx*x, sy*y, A*z+B, z)
  float4 lcol[4];                 // columns of light-clip-from-camera-relative (xyz; the last is the translation)
  float4 sun_dir;                 // xyz: toward the sun; w: glare strength
  float4 sun_col;                 // rgb: sun radiance
  float4 sky_top, sky_mid, sky_hor, ground;  // rgb gradient stops; ground = lower-hemisphere ambient
  float4 ground_col;              // rgb: ground-plane albedo
  float4 fog;                     // x: haze density (1/m), y: style (0 outdoor, 1 studio), z: edge width (px), w: ambient scale
  float4 vp;                      // viewport w, h; shadows enabled; shadow texel size in metres
  float4 camf;                    // camera position mod 10 (xy), z; 1/shadow-map size in w
  float4 wx;                      // weather: x overcast 0..1, y streak alpha, z streak length (m), w sim time (s); all 0 = clear
  float4 rainv;                   // drop velocity xyz (m/s, world; z < 0 falls), w: streak box edge (m)
  float4 rcam;                    // camera position mod rainv.w (xyz)
};
struct Draw {
  float4 rot[3];  // columns of world-from-geom rotation
  float4 trel;    // xyz: camera-relative translation (double-subtracted on the CPU)
  float4 color;
  float4 mat;     // specular, shininess, metallic, roughness (metallic / roughness < 0: unset)
  float4 flags;   // x: 1 = ground plane
};
[[vk::binding(0, 0)]] ConstantBuffer<Scene> sc;
[[vk::binding(1, 0)]] StructuredBuffer<Draw> draws;
[[vk::binding(2, 0)]] Texture2D<float> shadowTex;
[[vk::binding(3, 0)]] SamplerComparisonState shadowSmp;
struct Push {
  uint draw;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

static const float kPi = 3.14159265;

float3 to_lin(float3 c) { return pow(max(c, 0.0), 2.2); }
float3 to_srgb(float3 c) {
  c = saturate(c);
  return select(c <= 0.0031308, c * 12.92, 1.055 * pow(c, 1.0 / 2.4) - 0.055);
}

float4 clip_of(float3 rel) {
  const float3 v = float3(dot(sc.view_r.xyz, rel), dot(sc.view_u.xyz, rel), dot(sc.view_f.xyz, rel));
  return float4(sc.proj.x * v.x, sc.proj.y * v.y, sc.proj.z * v.z + sc.proj.w, v.z);
}

// ---- geometry --------------------------------------------------------------------------------------------------------
struct VsIn {
  float3 pos : POSITION;
  float3 nrm : NORMAL;
};
struct VsOut {
  float4 clip : SV_Position;
  float3 rel : TEXCOORD0;
  float3 n : NORMAL;
  float3 lp : TEXCOORD1;  // geom-frame position (terrain texturing)
};

[shader("vertex")]
VsOut vsMain(VsIn i) {
  const Draw d = draws[pc.draw];
  VsOut o;
  o.lp = i.pos;
  o.rel = d.rot[0].xyz * i.pos.x + d.rot[1].xyz * i.pos.y + d.rot[2].xyz * i.pos.z + d.trel.xyz;
  o.n = d.rot[0].xyz * i.nrm.x + d.rot[1].xyz * i.nrm.y + d.rot[2].xyz * i.nrm.z;
  o.clip = clip_of(o.rel);
  return o;
}

[shader("vertex")]
float4 vsShadow(VsIn i) : SV_Position {
  const Draw d = draws[pc.draw];
  const float3 rel = d.rot[0].xyz * i.pos.x + d.rot[1].xyz * i.pos.y + d.rot[2].xyz * i.pos.z + d.trel.xyz;
  return float4(sc.lcol[0].xyz * rel.x + sc.lcol[1].xyz * rel.y + sc.lcol[2].xyz * rel.z + sc.lcol[3].xyz, 1);
}

[shader("fragment")]
float fsLin(VsOut i) : SV_Target0 { return dot(sc.view_f.xyz, i.rel); }

float shadow_vis(float3 rel, float3 n) {
  if (sc.vp.z < 0.5) return 1.0;
  const float3 p = rel + n * (sc.vp.w * 1.5);  // normal offset: kills most acne without a big depth bias
  const float3 s = sc.lcol[0].xyz * p.x + sc.lcol[1].xyz * p.y + sc.lcol[2].xyz * p.z + sc.lcol[3].xyz;
  const float2 uv = s.xy * 0.5 + 0.5;
  if (any(uv < 0.0) || any(uv > 1.0) || s.z > 1.0) return 1.0;  // outside the fitted map: lit (not the border's "all shadow")
  const float px = sc.camf.w;
  float sum = 0;
  [unroll] for (int y = -1; y <= 1; ++y)
    [unroll] for (int x = -1; x <= 1; ++x) sum += shadowTex.SampleCmpLevelZero(shadowSmp, uv + float2(x, y) * px, s.z - 0.0005);
  return sum / 9.0;
}

// ---- sky -------------------------------------------------------------------------------------------------------------
float hash21(float2 p) {
  p = frac(p * float2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return frac(p.x * p.y);
}
float vnoise(float2 p) {
  const float2 i = floor(p), f = frac(p), u = f * f * (3.0 - 2.0 * f);
  return lerp(lerp(hash21(i), hash21(i + float2(1, 0)), u.x), lerp(hash21(i + float2(0, 1)), hash21(i + float2(1, 1)), u.x), u.y);
}
float fbm(float2 p) {
  float a = 0.5, s = 0;
  [unroll] for (int k = 0; k < 5; ++k) {
    s += a * vnoise(p);
    p = p * 2.03 + 17.0;
    a *= 0.5;
  }
  return s;
}

// Sky colour in a direction, without clouds / sun disc (also the reflection probe of shiny surfaces).
float3 sky_base(float3 dir) {
  const float e = dir.z;
  float3 c = lerp(sc.sky_hor.rgb, sc.sky_mid.rgb, smoothstep(0.0, 0.28, e));
  c = lerp(c, sc.sky_top.rgb, smoothstep(0.18, 0.95, e));
  return lerp(sc.sky_hor.rgb, c, smoothstep(-0.08, 0.0, e));  // below the horizon: the haze colour
}

[shader("vertex")]
float4 vsSky(uint vid : SV_VertexID) : SV_Position {
  const float2 p = float2(vid == 1 ? 3.0 : -1.0, vid == 2 ? 3.0 : -1.0);
  return float4(p, 1, 1);
}

[shader("fragment")]
float4 fsSky(float4 frag : SV_Position) : SV_Target0 {
  const float2 ndc = frag.xy / sc.vp.xy * 2.0 - 1.0;
  const float3 dir = normalize(sc.view_f.xyz + (ndc.x / sc.proj.x) * sc.view_r.xyz + (ndc.y / sc.proj.y) * sc.view_u.xyz);
  float3 c = sky_base(dir);
  if (sc.fog.y < 0.5) {
    const float3 L = sc.sun_dir.xyz;
    const float sd = saturate(dot(dir, L));
    c += float3(1.0, 0.72, 0.42) * (0.55 * pow(sd, 6.0) + 0.5 * pow(sd, 48.0)) * sc.sun_dir.w;  // warm glare
    c += float3(1.0, 0.95, 0.85) * smoothstep(0.9993, 0.9998, sd) * 4.0 * (1.0 - sc.wx.x);
    if (dir.z > 0.0) {  // clouds: fbm on a plane overhead, fading into the haze toward the horizon
      const float2 uv = dir.xy / (dir.z + 0.12) * 1.3 + sc.camf.xy * 0.004;
      const float n = fbm(uv);
      float cov = smoothstep(0.50, 0.80, n) * smoothstep(0.0, 0.16, dir.z);
      const float lit = saturate(0.55 + (n - fbm(uv + L.xy * 0.12)) * 3.0);
      float3 cc = lerp(float3(0.66, 0.70, 0.80), float3(1.0, 0.97, 0.93), lit);
      cc = lerp(sc.sky_hor.rgb, cc, smoothstep(0.0, 0.4, dir.z));  // low clouds take the horizon's warm glow
      // overcast (rain): a complete grey deck, darker where it is thick, lighter toward the horizon
      cov = lerp(cov, smoothstep(0.0, 0.08, dir.z), sc.wx.x);
      cc = lerp(cc, lerp(sc.sky_hor.rgb, sc.sky_mid.rgb * (0.78 + 0.5 * n), smoothstep(0.0, 0.5, dir.z)), sc.wx.x);
      c = lerp(c, cc, cov * lerp(0.92, 1.0, sc.wx.x));
    }
  }
  return float4(to_srgb(c), 1);
}

// ---- water -----------------------------------------------------------------------------------------------------------
// The sea mesh (flags.y): d.color is the deep-water colour. Body colour = deep colour lit by the sky and (a little) the sun, turning
// turquoise on the wave crests (z above the still water); Schlick Fresnel mixes in the sky reflection (strong at grazing view), the sun
// leaves a glitter (Blinn lobes), and fine static ripples (sines whose periods divide 10 m, so they tile with camf.xy) break up the mirror.
float4 water_shade(VsOut i, Draw d) {
  const float dist = length(i.rel);
  const float3 V = -i.rel / max(dist, 1e-3);
  const float2 w = i.rel.xy + sc.camf.xy;
  const float fade = saturate(1.0 - dist / 160.0);
  float2 g = float2(0, 0);
  [unroll] for (int k = 0; k < 4; ++k) {
    const float2 n = float2(k == 0 ? 3.0 : (k == 1 ? -5.0 : (k == 2 ? 7.0 : 11.0)), k == 0 ? 2.0 : (k == 1 ? 6.0 : (k == 2 ? -4.0 : 3.0)));
    const float ph = 6.2831853 * dot(n, w) / 10.0 + 1.7 * k;
    g += normalize(n) * cos(ph) * (0.030 / (1.0 + k));
  }
  float3 N = normalize(normalize(i.n) + float3(g.x, g.y, 0.0) * fade);
  if (dot(N, V) < 0.0) N = reflect(N, V);  // a microfacet facing away at grazing view
  const float3 L = sc.sun_dir.xyz;
  const float ndl = saturate(dot(N, L));
  const float3 deep = to_lin(max(d.color.rgb, float3(0.05, 0.30, 0.38))) * 2.0;  // never darker than a blue-green sea, whatever the scene's plane colour  // the colour is the water's depth tint seen through a metre or two of it, brighter than a painted surface of that colour
  const float zw = i.rel.z + sc.camf.z;
  float3 body_col = lerp(deep, to_lin(float3(0.10, 0.45, 0.42)), 0.55 * smoothstep(-0.1, 0.55, zw));
  body_col = lerp(body_col, to_lin(float3(0.20, 0.27, 0.29)), 0.75 * sc.wx.x);  // overcast (rain): a grey-green sea, no turquoise
  // the water keeps its own blue-green under any sky colour: the ambient is the sky's brightness, not its tint (a dusk sky would turn it brown)
  const float3 hemi_c = lerp(sc.ground.rgb, sc.sky_mid.rgb, 0.8) * sc.fog.w;
  const float3 hemi = float3(1, 1, 1) * (0.25 + 0.75 * dot(hemi_c, float3(0.3333, 0.3333, 0.3333)));
  float3 col = body_col * (hemi * 1.6 + sc.sun_col.rgb * ndl * 0.35);
  const float nv = saturate(dot(N, V));
  const float F = 0.02 + 0.98 * pow(1.0 - nv, 5.0);
  const float3 sky_r = lerp(sky_base(reflect(-V, N)), float3(0.30, 0.50, 0.72), 0.8) * sc.fog.w;  // a blue sky, not the warm horizon band of a dusk preset
  col = lerp(col, sky_r, 0.6 * F);
  const float3 H = normalize(L + V);
  const float nh = saturate(dot(N, H));
  col += sc.sun_col.rgb * (F + 0.04) * (pow(nh, 900.0) * 1.8 + pow(nh, 120.0) * 0.12) * ndl;
  const float haze = 1.0 - exp(-dist * sc.fog.x * 1.2);  // melts into the horizon like the ground, keeping some of its own colour
  col = lerp(col, lerp(sc.sky_hor.rgb, float3(0.03, 0.14, 0.20) * sc.fog.w * 3.0 + body_col * 1.5, lerp(0.75, 0.1, sc.wx.x)), saturate(haze) * lerp(0.7, 0.9, sc.wx.x));
  return float4(to_srgb(col), 1);
}

// ---- terrain ---------------------------------------------------------------------------------------------------------
// Heightfield albedo (flags.y = 2; flags.z / w = the local z range of the mesh): green meadow low, tundra higher, bare rock on steep
// slopes (a noise-wobbled threshold at about 15-30 degrees), snow on the highest, gentler ground. Two fbm octaves of the geom-frame
// position break the bands up. Stylised, not a land-cover model.
float3 terrain_albedo(float3 lp, float3 N, Draw d) {
  const float t = saturate((lp.z - d.flags.z) / max(d.flags.w - d.flags.z, 1.0));
  const float n = fbm(lp.xy * 0.012), f = vnoise(lp.xy * 0.35);
  const float s = 1.0 - saturate(N.z);
  float3 grass = to_lin(float3(0.26, 0.42, 0.18)), tundra = to_lin(float3(0.36, 0.42, 0.24)), rock = to_lin(float3(0.40, 0.40, 0.42));
  float3 a = lerp(grass, tundra, smoothstep(0.22, 0.60, t + (n - 0.5) * 0.25));
  a = lerp(a, rock * (0.7 + 0.5 * n), smoothstep(0.05, 0.14, s + (n - 0.5) * 0.06));
  const float snow = smoothstep(0.66, 0.86, t + (n - 0.5) * 0.3) * smoothstep(0.30, 0.62, fbm(lp.xy * 0.02 + 7.0)) * (1.0 - smoothstep(0.12, 0.22, s));  // patchy
  a = lerp(a, to_lin(float3(0.93, 0.95, 0.98)), snow);
  return a * (0.88 + 0.24 * f);
}

// ---- lit surfaces ----------------------------------------------------------------------------------------------------
[shader("fragment")]
float4 fsMain(VsOut i) : SV_Target0 {
  const Draw d = draws[pc.draw];
  if (d.flags.y > 0.5 && d.flags.y < 1.5) return water_shade(i, d);
  const float3 V = -normalize(i.rel);
  float3 N = normalize(i.n);
  const bool ground = d.flags.x > 0.5;
  if (ground && N.z * V.z < 0) N = -N;
  const bool terrain = d.flags.y > 1.5;
  float3 albedo = ground ? sc.ground_col.rgb : terrain ? terrain_albedo(i.lp, N, d) : to_lin(d.color.rgb);
  if (ground && sc.fog.y > 0.5) {  // studio floor: 1 m / 0.1 m grid, fading out
    const float2 w = i.rel.xy + sc.camf.xy;
    const float dist = length(i.rel);
    const float2 fw = max(fwidth(w), 1e-5);
    const float2 g1 = abs(frac(w - 0.5) - 0.5) / fw, g2 = abs(frac(w * 10.0 - 0.5) - 0.5) / (fw * 10.0);
    const float l1 = 1.0 - saturate(min(g1.x, g1.y) - 0.5), l2 = (1.0 - saturate(min(g2.x, g2.y) - 0.5)) * exp(-dist * 0.35);
    // 1 cm lines: only close to the camera, and only while a pixel is much smaller than the pitch (benchtop-scale designs)
    const float2 g3 = abs(frac(w * 100.0 - 0.5) - 0.5) / (fw * 100.0);
    const float l3 = (1.0 - saturate(min(g3.x, g3.y) - 0.5)) * exp(-dist * 3.0) * saturate(1.0 - max(fw.x, fw.y) * 80.0);
    albedo = lerp(albedo, albedo * 0.45, saturate(max(max(l1 * 0.9, l2 * 0.35), l3 * 0.16)));
  }
  const float metal = d.mat.z >= 0 ? d.mat.z : 0.0;
  const float3 L = sc.sun_dir.xyz;
  const float ndl = saturate(dot(N, L));
  const float vis = ndl > 0 ? shadow_vis(i.rel, N) : 1.0;
  const float3 hemi = lerp(sc.ground.rgb, sc.sky_mid.rgb, N.z * 0.5 + 0.5) * sc.fog.w;
  float3 col = albedo * (1.0 - metal) * (sc.sun_col.rgb * ndl * vis + hemi);
  // specular: Blinn-Phong lobe (exponent from shininess or roughness) x Schlick Fresnel; metals tint it with the albedo
  float shin = d.mat.y * 128.0;
  if (d.mat.w >= 0) shin = clamp(2.0 / max(d.mat.w * d.mat.w * d.mat.w * d.mat.w, 1e-4) - 2.0, 1.0, 1024.0);
  const float f0 = d.mat.x * 0.08;
  const float3 F0 = lerp(float3(f0, f0, f0), albedo, metal);
  const float3 H = normalize(L + V);
  const float3 F = F0 + (1.0 - F0) * pow(1.0 - saturate(dot(V, H)), 5.0);
  col += F * pow(saturate(dot(N, H)), shin) * (shin + 8.0) / (8.0 * kPi) * ndl * vis * sc.sun_col.rgb;
  if (!ground) col += F0 * sky_base(reflect(-V, N)) * (0.35 + 0.65 * metal) * sc.fog.w;  // sky reflection
  // distance haze toward the horizon colour
  const float dist = length(i.rel);
  // outdoor: the ground melts into the horizon within ~100 m while objects (drones at range) stay readable
  const float k = sc.fog.y > 0.5 ? 1.0 : (ground ? 4.0 : (terrain ? 0.2 : 0.5));  // terrain: distant ridges fade, not vanish
  const float haze = 1.0 - exp(-dist * sc.fog.x * k);
  col = lerp(col, sc.sky_hor.rgb, saturate(haze));
  return float4(to_srgb(col), 1);
}

// ---- edges -----------------------------------------------------------------------------------------------------------
// Vertex layout reuse: pos = one end, nrm = the other end of the segment; vid % 6 picks the quad corner.
float4 biased(float3 rel) {  // clip position pulled slightly toward the camera so lines win the depth test on their own faces
  float4 c = clip_of(rel);
  const float zb = c.w * 0.997 - 0.0004;
  c.z = (sc.proj.z + sc.proj.w / zb) * c.w;
  return c;
}
[shader("vertex")]
float4 vsEdge(VsIn i, uint vid : SV_VertexID) : SV_Position {
  const Draw d = draws[pc.draw];
  const float3 a = d.rot[0].xyz * i.pos.x + d.rot[1].xyz * i.pos.y + d.rot[2].xyz * i.pos.z + d.trel.xyz;
  const float3 b = d.rot[0].xyz * i.nrm.x + d.rot[1].xyz * i.nrm.y + d.rot[2].xyz * i.nrm.z + d.trel.xyz;
  float4 ca = biased(a), cb = biased(b);
  const uint k = vid % 6;
  const bool at_b = (k == 1 || k == 2 || k == 4);
  const float side = (k == 2 || k == 4 || k == 5) ? 1.0 : -1.0;
  // a segment crossing the near plane is pulled to the plane (cheap clip: keeps its direction)
  const float nearw = 0.06;
  if (ca.w < nearw && cb.w < nearw) return float4(0, 0, -2, 1);
  if (ca.w < nearw) ca = lerp(ca, cb, (nearw - ca.w) / (cb.w - ca.w));
  if (cb.w < nearw) cb = lerp(cb, ca, (nearw - cb.w) / (ca.w - cb.w));
  const float2 sa = ca.xy / ca.w * sc.vp.xy * 0.5, sb = cb.xy / cb.w * sc.vp.xy * 0.5;
  float2 dir = sb - sa;
  dir = length(dir) > 1e-6 ? normalize(dir) : float2(1, 0);
  const float2 off = float2(-dir.y, dir.x) * side * sc.fog.z / sc.vp.xy;  // half width in px -> ndc: px * 2 / vp * 0.5
  float4 c = at_b ? cb : ca;
  c.xy += off * c.w;
  return c;
}
[shader("fragment")]
float4 fsEdge() : SV_Target0 { return float4(to_srgb(float3(0.012, 0.014, 0.02)), 1); }

// ---- rain streaks ----------------------------------------------------------------------------------------------------
// One instance = one drop. Its start point is a hash of the instance id in a box of edge rainv.w; it moves with the drop velocity
// (wind drift + fall) and the box wraps around the camera, so the drops are fixed in the world (parallax and speed are right) and
// the streak count is the same everywhere. The streak is a screen-space quad of >= 1.4 px, from the drop to a point sc.wx.z metres
// up its path (the exposure length), pulled to the near plane when it crosses it.
struct RainOut {
  float4 clip : SV_Position;
  float a : TEXCOORD0;
};
uint pcg(uint v) {
  const uint s = v * 747796405u + 2891336453u;
  const uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
[shader("vertex")]
RainOut vsRain(uint vid : SV_VertexID, uint iid : SV_InstanceID) {
  RainOut o;
  const float edge = sc.rainv.w;
  const uint h0 = pcg(iid * 3u + 1u), h1 = pcg(h0 + 7u), h2 = pcg(h1 + 13u), h3 = pcg(h2 + 29u);
  const float3 b = float3(h0, h1, h2) * (edge / 4294967296.0);
  float3 r = b + sc.rainv.xyz * sc.wx.w - sc.rcam.xyz;
  r -= edge * floor(r / edge + 0.5);  // nearest periodic image to the camera
  const float3 dirv = normalize(sc.rainv.xyz);
  const float3 pa = r, pb = r - dirv * sc.wx.z * (0.7 + 0.6 * float(h3 >> 8u) / 16777216.0);
  float4 ca = clip_of(pa), cb = clip_of(pb);
  const uint k = vid % 6;
  const bool at_b = (k == 1 || k == 2 || k == 4);
  const float side = (k == 2 || k == 4 || k == 5) ? 1.0 : -1.0;
  const float nearw = 0.1;
  o.a = 0.0;
  o.clip = float4(0, 0, -2, 1);
  if (ca.w < nearw && cb.w < nearw) return o;
  if (ca.w < nearw) ca = lerp(ca, cb, (nearw - ca.w) / (cb.w - ca.w));
  if (cb.w < nearw) cb = lerp(cb, ca, (nearw - cb.w) / (ca.w - cb.w));
  const float2 sa = ca.xy / ca.w * sc.vp.xy * 0.5, sb = cb.xy / cb.w * sc.vp.xy * 0.5;
  float2 d2 = sb - sa;
  d2 = length(d2) > 1e-6 ? normalize(d2) : float2(0, 1);
  const float2 off = float2(-d2.y, d2.x) * side * 0.75 / sc.vp.xy * 2.0;  // 0.75 px half width -> ndc
  float4 c = at_b ? cb : ca;
  c.xy += off * c.w;
  o.clip = c;
  const float dist = length(pa);
  o.a = sc.wx.y * (0.45 + 0.55 * float(h3 & 255u) / 255.0) * smoothstep(0.5 * edge, 0.25 * edge, dist) * (at_b ? 0.35 : 1.0) * smoothstep(0.1, 0.8, dist);
  return o;
}
[shader("fragment")]
float4 fsRain(RainOut i) : SV_Target0 {
  const float3 col = lerp(sc.sky_hor.rgb, float3(0.92, 0.95, 1.0), 0.55);
  return float4(to_srgb(col), i.a);
}
