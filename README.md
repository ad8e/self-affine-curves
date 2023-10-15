An attempt at finding an optimal class of curves for https://ad8e.pages.dev/curve

Not looking promising, because of curvature issues...discussion going on at https://xi.zulipchat.com/#narrow/stream/260979-kurbo/topic/curves.20w.2F.20perspective.20transform

This repository is missing boring parts like the build script and output header. If it becomes interesting, it'll become worth my time to fix up the build steps here. Temporary demo at https://ad8e.pages.dev/curve2

Uses Ceres to solve the curvature equations. Emscripten compiles the C++ to WASM so I can run it on a webpage.