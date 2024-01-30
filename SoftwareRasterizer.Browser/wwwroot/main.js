// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

import { dotnet } from './dotnet.js';

const is_browser = typeof window != "undefined";
if (!is_browser) throw new Error(`Expected to be running in a browser`);

const { setModuleImports, getAssemblyExports, getConfig, runMainAndExit } = await dotnet
    .withDiagnosticTracing(false)
    .withApplicationArgumentsFromQuery()
    .withDebugging(1)
    .withConfig({
        assets: [
            {
                "behavior": "vfs",
                "name": "Castle/IndexBuffer.bin",
                "isOptional": true,
            },
            {
                "behavior": "vfs",
                "name": "Castle/VertexBuffer.bin",
                "isOptional": true,
            },
        ]
    })
    .create();

const canvas = document.getElementById("outCanvas");
const canvasCtx = canvas.getContext("bitmaprenderer");

setModuleImports("main.js", {
    window: {
        requestAnimationFrame: () => globalThis.window.requestAnimationFrame
    }
});

const config = getConfig();
const exports = await getAssemblyExports(config.mainAssemblyName);

await runMainAndExit(config.mainAssemblyName, [canvas.width.toString(), canvas.height.toString()]);