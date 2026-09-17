/**
 * DOCSHIELD Three.js visualizations — intentional, disposable, reduced-motion aware.
 * CDN: three@0.160.0
 */
(function (global) {
  "use strict";

  const THREE = global.THREE;
  if (!THREE) {
    console.warn("Three.js not loaded — 3D visualizations disabled.");
    global.DocshieldThree = {
      ready: false,
      initHero() {},
      initUpload() {},
      initScanner() {},
      initGraph() {},
      initIdentity() {},
      setScanActive() {},
      setGraphData() {},
      disposeAll() {},
    };
    return;
  }

  const reduced =
    global.matchMedia && global.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const lowGfx = !!(global.localStorage && global.localStorage.getItem("docshield_low_gfx") === "1");

  const scenes = [];

  function makeRenderer(canvas) {
    const renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: !lowGfx,
      alpha: true,
      powerPreference: "high-performance",
    });
    renderer.setPixelRatio(Math.min(global.devicePixelRatio || 1, lowGfx ? 1 : 1.75));
    renderer.setClearColor(0x000000, 0);
    return renderer;
  }

  function fit(renderer, camera, el) {
    const w = el.clientWidth || 1;
    const h = el.clientHeight || 1;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  function disposeObject(obj) {
    obj.traverse((child) => {
      if (child.geometry) child.geometry.dispose();
      if (child.material) {
        const mats = Array.isArray(child.material) ? child.material : [child.material];
        mats.forEach((m) => {
          if (m.map) m.map.dispose();
          m.dispose();
        });
      }
    });
  }

  function createDocPlane(color = 0x2a3a4a) {
    const geo = new THREE.PlaneGeometry(1.4, 1.0, 1, 1);
    const mat = new THREE.MeshStandardMaterial({
      color,
      metalness: 0.15,
      roughness: 0.55,
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geo, mat);
    const frame = new THREE.LineSegments(
      new THREE.EdgesGeometry(geo),
      new THREE.LineBasicMaterial({ color: 0x3a7ca5 })
    );
    mesh.add(frame);
    return mesh;
  }

  function createParticles(count) {
    const n = lowGfx ? Math.min(count, 40) : count;
    const positions = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      positions[i * 3] = (Math.random() - 0.5) * 4;
      positions[i * 3 + 1] = (Math.random() - 0.5) * 3;
      positions[i * 3 + 2] = (Math.random() - 0.5) * 2;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const mat = new THREE.PointsMaterial({
      color: 0x3a7ca5,
      size: 0.03,
      transparent: true,
      opacity: 0.7,
    });
    return new THREE.Points(geo, mat);
  }

  function mountScene(canvas, setup) {
    if (!canvas) return null;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 50);
    camera.position.set(0, 0.2, 3.2);
    const renderer = makeRenderer(canvas);
    const light = new THREE.DirectionalLight(0xffffff, 1.1);
    light.position.set(2, 3, 4);
    scene.add(light);
    scene.add(new THREE.AmbientLight(0x405060, 0.7));

    const ctx = { scene, camera, renderer, canvas, objects: {}, active: true, t: 0 };
    setup(ctx);
    fit(renderer, camera, canvas.parentElement || canvas);

    function onResize() {
      fit(renderer, camera, canvas.parentElement || canvas);
    }
    global.addEventListener("resize", onResize);

    let raf = 0;
    function tick() {
      if (!ctx.active) return;
      raf = requestAnimationFrame(tick);
      if (document.hidden) return;
      ctx.t += reduced ? 0 : 0.01;
      if (ctx.update) ctx.update(ctx.t);
      renderer.render(scene, camera);
    }
    if (!reduced) tick();
    else renderer.render(scene, camera);

    ctx.dispose = () => {
      ctx.active = false;
      cancelAnimationFrame(raf);
      global.removeEventListener("resize", onResize);
      disposeObject(scene);
      renderer.dispose();
    };
    scenes.push(ctx);
    return ctx;
  }

  function initHero(canvas) {
    return mountScene(canvas, (ctx) => {
      const doc = createDocPlane(0x1a2834);
      ctx.scene.add(doc);
      const pts = createParticles(90);
      ctx.scene.add(pts);
      const beam = new THREE.Mesh(
        new THREE.PlaneGeometry(1.5, 0.02),
        new THREE.MeshBasicMaterial({ color: 0x3a7ca5, transparent: true, opacity: 0.55 })
      );
      beam.position.z = 0.05;
      doc.add(beam);
      ctx.objects = { doc, pts, beam };
      ctx.update = (t) => {
        doc.rotation.y = Math.sin(t * 0.4) * 0.35;
        doc.rotation.x = 0.15 + Math.sin(t * 0.25) * 0.05;
        beam.position.y = Math.sin(t * 1.2) * 0.4;
        pts.rotation.y = t * 0.08;
      };
    });
  }

  function initUpload(canvas) {
    return mountScene(canvas, (ctx) => {
      const doc = createDocPlane(0x15202a);
      ctx.scene.add(doc);
      cameraAdjust(ctx.camera);
      ctx.update = (t) => {
        doc.rotation.y = t * 0.35;
      };
    });
  }

  function cameraAdjust(camera) {
    camera.position.set(0, 0, 2.6);
  }

  let scannerCtx = null;
  function initScanner(canvas) {
    scannerCtx = mountScene(canvas, (ctx) => {
      const doc = createDocPlane(0x182430);
      ctx.scene.add(doc);
      const beam = new THREE.Mesh(
        new THREE.PlaneGeometry(1.45, 0.025),
        new THREE.MeshBasicMaterial({ color: 0x2f9e6b, transparent: true, opacity: 0.7 })
      );
      beam.position.z = 0.06;
      doc.add(beam);
      const boxes = [];
      for (let i = 0; i < 4; i++) {
        const b = new THREE.LineSegments(
          new THREE.EdgesGeometry(new THREE.PlaneGeometry(0.28, 0.12)),
          new THREE.LineBasicMaterial({ color: i === 0 ? 0x2f9e6b : 0x3a7ca5 })
        );
        b.position.set(-0.4 + i * 0.28, 0.25 - (i % 2) * 0.35, 0.07);
        b.visible = false;
        doc.add(b);
        boxes.push(b);
      }
      ctx.objects = { doc, beam, boxes, scanning: false };
      ctx.update = (t) => {
        if (ctx.objects.scanning) {
          ctx.objects.beam.position.y = Math.sin(t * 2.2) * 0.42;
          ctx.objects.boxes.forEach((b, i) => {
            b.visible = true;
            b.material.opacity = 0.5 + 0.5 * Math.sin(t * 2 + i);
          });
        } else {
          ctx.objects.beam.position.y = 0;
        }
        doc.rotation.y = Math.sin(t * 0.2) * 0.12;
      };
    });
    return scannerCtx;
  }

  function setScanActive(on) {
    if (scannerCtx && scannerCtx.objects) scannerCtx.objects.scanning = !!on;
  }

  let graphCtx = null;
  function initGraph(canvas) {
    graphCtx = mountScene(canvas, (ctx) => {
      ctx.camera.position.set(0, 0.4, 5);
      const group = new THREE.Group();
      ctx.scene.add(group);
      ctx.objects = { group, nodes: {}, links: [] };
      ctx.update = (t) => {
        group.rotation.y = reduced ? 0 : t * 0.12;
      };
    });
    return graphCtx;
  }

  function setGraphData(labels) {
    if (!graphCtx) return;
    const { group } = graphCtx.objects;
    while (group.children.length) {
      const c = group.children.pop();
      disposeObject(c);
    }
    const names = labels && labels.length ? labels : [
      "CASE", "DOC", "OCR", "MRZ", "FORENSICS", "EffNet", "ViT", "CLIP", "FACE", "LIVE", "FUSION",
    ];
    const nodes = {};
    names.forEach((name, i) => {
      const angle = (i / names.length) * Math.PI * 2;
      const r = name === "CASE" ? 0 : 1.6;
      const mesh = new THREE.Mesh(
        new THREE.SphereGeometry(name === "CASE" ? 0.14 : 0.08, 12, 12),
        new THREE.MeshStandardMaterial({
          color: name === "CASE" ? 0x3a7ca5 : 0x2a4050,
          emissive: name === "CASE" ? 0x123040 : 0x000000,
          metalness: 0.2,
          roughness: 0.5,
        })
      );
      mesh.position.set(Math.cos(angle) * r, Math.sin(angle) * r * 0.55, Math.sin(angle) * 0.2);
      mesh.userData.label = name;
      group.add(mesh);
      nodes[name] = mesh;
      if (name !== "CASE" && nodes.CASE) {
        const pts = new Float32Array([
          nodes.CASE.position.x, nodes.CASE.position.y, nodes.CASE.position.z,
          mesh.position.x, mesh.position.y, mesh.position.z,
        ]);
        const geo = new THREE.BufferGeometry();
        geo.setAttribute("position", new THREE.BufferAttribute(pts, 3));
        const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: 0x2a4a5a, transparent: true, opacity: 0.55 }));
        group.add(line);
      }
    });
    graphCtx.objects.nodes = nodes;
  }

  function initIdentity(canvas) {
    return mountScene(canvas, (ctx) => {
      ctx.camera.position.set(0, 0, 3.4);
      const left = createDocPlane(0x1a3028);
      left.scale.set(0.55, 0.7, 1);
      left.position.x = -0.95;
      const right = createDocPlane(0x1a2834);
      right.scale.set(0.55, 0.7, 1);
      right.position.x = 0.95;
      const bridge = new THREE.Line(
        new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(-0.55, 0, 0),
          new THREE.Vector3(0.55, 0, 0),
        ]),
        new THREE.LineBasicMaterial({ color: 0x3a7ca5 })
      );
      const mid = new THREE.Mesh(
        new THREE.SphereGeometry(0.07, 10, 10),
        new THREE.MeshStandardMaterial({ color: 0x3a7ca5, emissive: 0x102030 })
      );
      ctx.scene.add(left, right, bridge, mid);
      ctx.update = (t) => {
        mid.position.y = Math.sin(t * 2) * 0.08;
        left.rotation.y = -0.2 + Math.sin(t) * 0.05;
        right.rotation.y = 0.2 + Math.sin(t + 1) * 0.05;
      };
    });
  }

  function disposeAll() {
    while (scenes.length) {
      const s = scenes.pop();
      if (s && s.dispose) s.dispose();
    }
  }

  function resizeAll() {
    scenes.forEach((ctx) => {
      if (!ctx || !ctx.renderer || !ctx.camera) return;
      fit(ctx.renderer, ctx.camera, ctx.canvas.parentElement || ctx.canvas);
      if (ctx.active) ctx.renderer.render(ctx.scene, ctx.camera);
    });
  }

  global.DocshieldThree = {
    ready: true,
    reduced,
    lowGfx,
    initHero,
    initUpload,
    initScanner,
    initGraph,
    initIdentity,
    setScanActive,
    setGraphData,
    resizeAll,
    disposeAll,
  };
})(window);
