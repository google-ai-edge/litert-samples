// This code implements the `-sMODULARIZE` settings by taking the generated
// JS program code (INNER_JS_CODE) and wrapping it in a factory function.

// Single threaded MINIMAL_RUNTIME programs do not need access to
// document.currentScript, so a simple export declaration is enough.
var ModuleFactory = (() => {
  // When MODULARIZE this JS may be executed later,
  // after document.currentScript is gone, so we save it.
  // In EXPORT_ES6 mode we can just use 'import.meta.url'.
  var _scriptName = globalThis.document?.currentScript?.src;
  return async function(moduleArg = {}) {
    var Module = moduleArg;
    // include: shell.js
    // include: minimum_runtime_check.js
    // end include: minimum_runtime_check.js
    // The Module object: Our interface to the outside world. We import
    // and export values on it. There are various ways Module can be used:
    // 1. Not defined. We create it here
    // 2. A function parameter, function(moduleArg) => Promise<Module>
    // 3. pre-run appended it, var Module = {}; ..generated code..
    // 4. External script tag defines var Module.
    // We need to check if Module already exists (e.g. case 3 above).
    // Substitution will be replaced with actual code on later stage of the
    // build, this way Closure Compiler will not mangle it (e.g. case 4. above).
    // Note that if you want to run closure, and also to use Module
    // after the generated code, you will need to define   var Module = {};
    // before the code. Then that object will be used in the code, and you
    // can continue to use Module afterwards as well.
    // Determine the runtime environment we are in. You can customize this by
    // setting the ENVIRONMENT setting at compile time (see settings.js).
    // Attempt to auto-detect the environment
    var ENVIRONMENT_IS_WEB = !!globalThis.window;

    var ENVIRONMENT_IS_WORKER = !!globalThis.WorkerGlobalScope;

    // N.b. Electron.js environment is simultaneously a NODE-environment, but
    // also a web environment.
    var ENVIRONMENT_IS_NODE = globalThis.process?.versions?.node &&
        globalThis.process?.type != 'renderer';

    // --pre-jses are emitted after the Module integration code, so that they
    // can refer to Module (if they choose; they can also define Module)
    var programArgs = [];

    var thisProgram = './this.program';

    var quit_ = (status, toThrow) => {
      throw toThrow;
    };

    if (typeof __filename != 'undefined') {
      // Node
      _scriptName = __filename;
    } else if (ENVIRONMENT_IS_WORKER) {
      _scriptName = self.location.href;
    }

    // `/` should be present at the end if `scriptDirectory` is not empty
    var scriptDirectory = '';

    function locateFile(path) {
      if (Module['locateFile']) {
        return Module['locateFile'](path, scriptDirectory);
      }
      return scriptDirectory + path;
    }

    // Hooks that are implemented differently in different runtime environments.
    var readAsync, readBinary;

    if (ENVIRONMENT_IS_NODE) {
      // These modules will usually be used on Node.js. Load them eagerly to
      // avoid the complexity of lazy-loading.
      var fs = require('node:fs');
      scriptDirectory = __dirname + '/';
      // include: node_shell_read.js
      readBinary = filename => {
        // We need to re-wrap `file://` strings to URLs.
        filename = isFileURI(filename) ? new URL(filename) : filename;
        var ret = fs.readFileSync(filename);
        return ret;
      };
      readAsync = async (filename, binary = true) => {
        // See the comment in the `readBinary` function.
        filename = isFileURI(filename) ? new URL(filename) : filename;
        var ret = fs.readFileSync(filename, binary ? undefined : 'utf8');
        return ret;
      };
      // end include: node_shell_read.js
      if (process.argv.length > 1) {
        thisProgram = process.argv[1].replace(/\\/g, '/');
      }
      programArgs = process.argv.slice(2);
      quit_ = (status, toThrow) => {
        process.exitCode = status;
        throw toThrow;
      };
    } else  // Note that this includes Node.js workers when relevant (pthreads
            // is enabled).
      // Node.js workers are detected as a combination of ENVIRONMENT_IS_WORKER
      // and ENVIRONMENT_IS_NODE.
      if (ENVIRONMENT_IS_WEB || ENVIRONMENT_IS_WORKER) {
        try {
          scriptDirectory = new URL('.', _scriptName).href;
        } catch {
        }
        {
          // include: web_or_worker_shell_read.js
          if (ENVIRONMENT_IS_WORKER) {
            readBinary = url => {
              var xhr = new XMLHttpRequest;
              xhr.open('GET', url, false);
              xhr.responseType = 'arraybuffer';
              xhr.send(null);
              return new Uint8Array(/** @type{!ArrayBuffer} */ (xhr.response));
            };
          }
          readAsync = async url => {
            // Fetch has some additional restrictions over XHR, like it can't be
            // used on a file:// url. See
            // https://github.com/github/fetch/pull/92#issuecomment-140665932
            // Cordova or Electron apps are typically loaded from a file:// url.
            // So use XHR on webview if URL is a file URL.
            if (isFileURI(url)) {
              return new Promise((resolve, reject) => {
                var xhr = new XMLHttpRequest;
                xhr.open('GET', url, true);
                xhr.responseType = 'arraybuffer';
                xhr.onload = () => {
                  if (xhr.status == 200 || (xhr.status == 0 && xhr.response)) {
                    // file URLs can return 0
                    resolve(xhr.response);
                    return;
                  }
                  reject(xhr.status);
                };
                xhr.onerror = reject;
                xhr.send(null);
              });
            }
            var response = await fetch(url, {credentials: 'same-origin'});
            if (response.ok) {
              return response.arrayBuffer();
            }
            throw new Error(response.status + ' : ' + response.url);
          };
        }
      } else {
      }

    var out = console.log.bind(console);

    var err = console.error.bind(console);

    // end include: shell.js
    // include: preamble.js
    // === Preamble library stuff ===
    // Documentation for the public APIs defined in this file must be updated
    // in:
    //    site/source/docs/api_reference/preamble.js.rst
    // A prebuilt local version of the documentation is available at:
    //    site/build/text/docs/api_reference/preamble.js.txt
    // You can also build docs locally as HTML or other formats in site/
    // An online HTML version (which may be of a different version of
    // Emscripten)
    //    is up at
    //    http://kripken.github.io/emscripten-site/docs/api_reference/preamble.js.html
    var wasmBinary;

    // Wasm globals
    //========================================
    // Runtime essentials
    //========================================
    // whether we are quitting the application. no code should run after this.
    // set in exit() and abort()
    var ABORT = false;

    // set by exit() and abort().  Passed to 'onExit' handler.
    // NOTE: This is also used as the process return code in shell environments
    // but only when noExitRuntime is false.
    var EXITSTATUS;

    // In STRICT mode, we only define assert() when ASSERTIONS is set.  i.e. we
    // don't define it at all in release modes.  This matches the behaviour of
    // MINIMAL_RUNTIME.
    // TODO(sbc): Make this the default even without STRICT enabled.
    /** @type {function(*, string=)} */ function assert(condition, text) {
      if (!condition) {
        // This build was created without ASSERTIONS defined.  `assert()` should
        // not ever be called in this configuration but in case there are
        // callers in the wild leave this simple abort() implementation here for
        // now.
        abort(text);
      }
    }

    /**
     * Indicates whether filename is delivered via file protocol (as opposed to
     * http/https)
     * @noinline
     */
    var isFileURI = filename => filename.startsWith('file://');

    // include: runtime_common.js
    // include: runtime_exceptions.js
    // Base Emscripten EH error class
    class EmscriptenEH {}

    class EmscriptenSjLj extends EmscriptenEH {}

    // end include: runtime_exceptions.js
    // include: runtime_debug.js
    // end include: runtime_debug.js
    // Memory management
    var runtimeInitialized = false;

    // When ALLOW_MEMORY_GROWTH is enabled, the conversion from Wasm
    // memory to ArrayBuffer requires some additional logic.
    function getMemoryBuffer() {
      return wasmMemory.buffer;
    }

    function updateMemoryViews() {
      // If we already have a heap that is resizeable/growable buffer we don't
      // need to do anything in updateMemoryViews.
      if (HEAP8?.buffer?.resizable) return;
      var b = getMemoryBuffer();
      HEAP8 = new Int8Array(b);
      HEAP16 = new Int16Array(b);
      Module['HEAPU8'] = HEAPU8 = new Uint8Array(b);
      HEAPU16 = new Uint16Array(b);
      HEAP32 = new Int32Array(b);
      Module['HEAPU32'] = HEAPU32 = new Uint32Array(b);
      Module['HEAPF32'] = HEAPF32 = new Float32Array(b);
      Module['HEAPF64'] = HEAPF64 = new Float64Array(b);
    }

    // include: memoryprofiler.js
    // end include: memoryprofiler.js
    // end include: runtime_common.js
    function preRun() {
      var preRun = Module['preRun'];
      if (preRun) {
        if (typeof preRun == 'function') preRun = [preRun];
        onPreRuns.push(...preRun);
      }
      // Begin ATPRERUNS hooks
      callRuntimeCallbacks(onPreRuns);
    }

    function initRuntime() {
      runtimeInitialized = true;
      // Begin ATINITS hooks
      if (!Module['noFSInit'] && !FS.initialized) FS.init();
      TTY.init();
      // End ATINITS hooks
      wasmExports['Fc']();
      // Begin ATPOSTCTORS hooks
      FS.ignorePermissions = false;
    }

    function postRun() {
      var postRun = Module['postRun'];
      if (postRun) {
        if (typeof postRun == 'function') postRun = [postRun];
        onPostRuns.push(...postRun);
      }
      // Begin ATPOSTRUNS hooks
      callRuntimeCallbacks(onPostRuns);
    }

    /**
     * @param {string|number=} what
     */
    function abort(what) {
      Module['onAbort']?.(what);
      what = `Aborted(${what})`;
      // TODO(sbc): Should we remove printing and leave it up to whoever
      // catches the exception?
      err(what);
      ABORT = true;
      what += '. Build with -sASSERTIONS for more info.';
      // Use a wasm runtime error, because a JS error might be seen as a foreign
      // exception, which means we'd run destructors on it. We need the error to
      // simply make the program stop.
      // FIXME This approach does not work in Wasm EH because it currently does
      // not assume all RuntimeErrors are from traps; it decides whether a
      // RuntimeError is from a trap or not based on a hidden field within the
      // object. So at the moment we don't have a way of throwing a wasm trap
      // from JS. TODO Make a JS API that allows this in the wasm spec. Suppress
      // closure compiler warning here. Closure compiler's builtin extern
      // definition for WebAssembly.RuntimeError claims it takes no arguments
      // even though it can.
      // TODO(https://github.com/google/closure-compiler/pull/3913): Remove
      // if/when upstream closure gets fixed.
      /** @suppress {checkTypes} */ var e = new WebAssembly.RuntimeError(what);
      // Throw the error whether or not MODULARIZE is set because abort is used
      // in code paths apart from instantiation where an exception is expected
      // to be thrown when abort is called.
      throw e;
    }

    var wasmBinaryFile;

    function findWasmBinary() {
      return locateFile('genai_wasm_internal.wasm');
    }

    function getBinarySync(file) {
      if (readBinary) {
        return readBinary(file);
      }
      // Throwing a plain string here, even though it not normally advisable
      // since this gets turning into an `abort` in instantiateArrayBuffer.
      throw 'both async and sync fetching of the wasm failed';
    }

    async function getWasmBinary(binaryFile) {
      // If we don't have the binary yet, load it asynchronously using
      // readAsync.
      if (!wasmBinary) {
        // Fetch the binary using readAsync
        try {
          var response = await readAsync(binaryFile);
          return new Uint8Array(response);
        } catch {
        }
      }
      // Otherwise, getBinarySync should be able to get it synchronously
      return getBinarySync(binaryFile);
    }

    async function instantiateArrayBuffer(binaryFile, imports) {
      try {
        var binary = await getWasmBinary(binaryFile);
        var instance = await WebAssembly.instantiate(binary, imports);
        return instance;
      } catch (reason) {
        err(`failed to asynchronously prepare wasm: ${reason}`);
        abort(reason);
      }
    }

    async function instantiateAsync(binary, binaryFile, imports) {
      if (!binary && !isFileURI(binaryFile) && !ENVIRONMENT_IS_NODE) {
        try {
          var response = fetch(binaryFile, {credentials: 'same-origin'});
          var instantiationResult =
              await WebAssembly.instantiateStreaming(response, imports);
          return instantiationResult;
        } catch (reason) {
          // We expect the most common failure cause to be a bad MIME type for
          // the binary, in which case falling back to ArrayBuffer instantiation
          // should work.
          err(`wasm streaming compile failed: ${reason}`);
          err('falling back to ArrayBuffer instantiation');
        }
      }
      return instantiateArrayBuffer(binaryFile, imports);
    }

    function getWasmImports() {
      // prepare imports
      var imports = {'a': wasmImports};
      return imports;
    }

    // Create the wasm instance.
    // Receives the wasm imports, returns the exports.
    async function createWasm() {
      // Load the wasm module and create an instance of using native support in
      // the JS engine. handle a generated wasm instance, receiving its exports
      // and performing other necessary setup
      function receiveInstance(instance) {
        wasmExports = instance.exports;
        wasmExports = Asyncify.instrumentWasmExports(wasmExports);
        wasmExports = applySignatureConversions(wasmExports);
        assignWasmExports(wasmExports);
        updateMemoryViews();
        return wasmExports;
      }
      // Prefer streaming instantiation if available.
      function receiveInstantiationResult(result) {
        // 'result' is a ResultObject object which has both the module and
        // instance. receiveInstance() will swap in the exports (to Module.asm)
        // so they can be called
        // TODO: Due to Closure regression
        // https://github.com/google/closure-compiler/issues/3193, the above
        // line no longer optimizes out down to the following line. When the
        // regression is fixed, can restore the above PTHREADS-enabled path.
        return receiveInstance(result['instance']);
      }
      var info = getWasmImports();
      // User shell pages can write their own Module.instantiateWasm =
      // function(imports, successCallback) callback to manually instantiate the
      // Wasm module themselves. This allows pages to run the instantiation
      // parallel to any other async startup actions they are performing. Also
      // pthreads and wasm workers initialize the wasm instance through this
      // path.
      var instantiateWasm = Module['instantiateWasm'];
      if (instantiateWasm) {
        return new Promise(resolve => {
          instantiateWasm(info, inst => resolve(receiveInstance(inst)));
        });
      }
      wasmBinaryFile ??= findWasmBinary();
      var result = await instantiateAsync(wasmBinary, wasmBinaryFile, info);
      var exports = receiveInstantiationResult(result);
      return exports;
    }

    // Globals used by JS i64 conversions (see makeSetValue)
    var tempDouble;

    var tempI64;

    // end include: preamble.js
    // Begin JS library code
    var handleException = e => {
      // Certain exception types we do not treat as errors since they are used
      // for internal control flow.
      // 1. ExitStatus, which is thrown by exit()
      // 2. "unwind", which is thrown by emscripten_unwind_to_js_event_loop()
      // and others
      //    that wish to return to JS event loop.
      if (e instanceof ExitStatus || e == 'unwind') {
        return EXITSTATUS;
      }
      quit_(1, e);
    };

    class ExitStatus {
      name = 'ExitStatus';
      constructor(status) {
        this.message = `Program terminated with exit(${status})`;
        this.status = status;
      }
    }

    var runtimeKeepaliveCounter = 0;

    var keepRuntimeAlive = () => noExitRuntime || runtimeKeepaliveCounter > 0;

    var _proc_exit = code => {
      EXITSTATUS = code;
      if (!keepRuntimeAlive()) {
        Module['onExit']?.(code);
        ABORT = true;
      }
      quit_(code, new ExitStatus(code));
    };

    /** @param {boolean|number=} implicit */ var exitJS =
        (status, implicit) => {
          EXITSTATUS = status;
          _proc_exit(status);
        };

    var _exit = exitJS;

    var maybeExit = () => {
      if (!keepRuntimeAlive()) {
        try {
          _exit(EXITSTATUS);
        } catch (e) {
          handleException(e);
        }
      }
    };

    var callUserCallback = func => {
      if (ABORT) {
        return;
      }
      try {
        return func();
      } catch (e) {
        handleException(e);
      } finally {
        maybeExit();
      }
    };

    function getFullscreenElement() {
      return document.fullscreenElement ?? document.webkitFullscreenElement;
    }

    /** @param {number=} timeout */ var safeSetTimeout = (func, timeout) =>
        setTimeout(() => {
          callUserCallback(func);
        }, timeout);

    var warnOnce = text => {
      warnOnce.shown ||= {};
      if (!warnOnce.shown[text]) {
        warnOnce.shown[text] = 1;
        if (ENVIRONMENT_IS_NODE) text = 'warning: ' + text;
        err(text);
      }
    };

    var preloadPlugins = [];

    /** @type {!Int32Array} */ var HEAP32;

    /** @type {!Uint32Array} */ var HEAPU32;

    var Browser = {
      useWebGL: false,
      isFullscreen: false,
      pointerLock: false,
      moduleContextCreatedCallbacks: [],
      preloadedImages: {},
      preloadedAudios: {},
      getCanvas: () => Module['canvas'],
      init() {
        if (Browser.initted) return;
        Browser.initted = true;
        // Support for plugins that can process preloaded files. You can add
        // more of these to your app by creating and appending to
        // preloadPlugins. Each plugin is asked if it can handle a file based on
        // the file's name. If it can, it is given the file's raw data. When it
        // is done, it calls a callback with the file's (possibly modified)
        // data. For example, a plugin might decompress a file, or it might
        // create some side data structure for use later (like an Image element,
        // etc.).
        var imagePlugin = {};
        imagePlugin['canHandle'] = name => !Module['noImageDecoding'] &&
            /\.(jpg|jpeg|png|bmp|webp)$/i.test(name);
        imagePlugin['handle'] = async (byteArray, name) => {
          var b = new Blob([byteArray], {type: Browser.getMimetype(name)});
          if (b.size !== byteArray.length) {
            // Safari bug #118630
            // Safari's Blob can only take an ArrayBuffer
            b = new Blob(
                [(new Uint8Array(byteArray)).buffer],
                {type: Browser.getMimetype(name)});
          }
          var url = URL.createObjectURL(b);
          return new Promise((resolve, reject) => {
            var img = new Image;
            img.onload = () => {
              var canvas = /** @type {!HTMLCanvasElement} */ (
                  document.createElement('canvas'));
              canvas.width = img.width;
              canvas.height = img.height;
              var ctx = canvas.getContext('2d');
              ctx.drawImage(img, 0, 0);
              Browser.preloadedImages[name] = canvas;
              URL.revokeObjectURL(url);
              resolve(byteArray);
            };
            img.onerror = event => {
              err(`Image ${url} could not be decoded`);
              reject();
            };
            img.src = url;
          });
        };
        preloadPlugins.push(imagePlugin);
        var audioPlugin = {};
        audioPlugin['canHandle'] = name => !Module['noAudioDecoding'] &&
            name.slice(-4) in {'.ogg': 1, '.wav': 1, '.mp3': 1};
        audioPlugin['handle'] =
            async (byteArray, name) => new Promise((resolve, reject) => {
          var done = false;
          function finish(audio) {
            if (done) return;
            done = true;
            Browser.preloadedAudios[name] = audio;
            resolve(byteArray);
          }
          var b = new Blob([byteArray], {type: Browser.getMimetype(name)});
          var url = URL.createObjectURL(b);
          // XXX we never revoke this!
          var audio = new Audio;
          audio.addEventListener('canplaythrough', () => finish(audio));
          // use addEventListener due to chromium bug 124926
          audio.onerror = event => {
            if (done) return;
            err(`warning: browser could not fully decode audio ${
                name}, trying slower base64 approach`);
            function encode64(data) {
              var BASE =
                  'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
              var PAD = '=';
              var ret = '';
              var leftchar = 0;
              var leftbits = 0;
              for (var i = 0; i < data.length; i++) {
                leftchar = (leftchar << 8) | data[i];
                leftbits += 8;
                while (leftbits >= 6) {
                  var curr = (leftchar >> (leftbits - 6)) & 63;
                  leftbits -= 6;
                  ret += BASE[curr];
                }
              }
              if (leftbits == 2) {
                ret += BASE[(leftchar & 3) << 4];
                ret += PAD + PAD;
              } else if (leftbits == 4) {
                ret += BASE[(leftchar & 15) << 2];
                ret += PAD;
              }
              return ret;
            }
            audio.src = 'data:audio/x-' + name.slice(-3) + ';base64,' +
                encode64(byteArray);
            finish(audio);
          };
          audio.src = url;
          // workaround for chrome bug 124926 - we do not always get
          // oncanplaythrough or onerror
          safeSetTimeout(() => {
            finish(audio);
          }, 1e4);
        });
        preloadPlugins.push(audioPlugin);
        // Canvas event setup
        function pointerLockChange() {
          var canvas = Browser.getCanvas();
          Browser.pointerLock = document.pointerLockElement === canvas;
        }
        var canvas = Browser.getCanvas();
        if (canvas) {
          // forced aspect ratio can be enabled by defining 'forcedAspectRatio'
          // on Module Module['forcedAspectRatio'] = 4 / 3;
          document.addEventListener('pointerlockchange', pointerLockChange);
          if (Module['elementPointerLock']) {
            canvas.addEventListener('click', ev => {
              if (!Browser.pointerLock &&
                  Browser.getCanvas().requestPointerLock) {
                Browser.getCanvas().requestPointerLock();
                ev.preventDefault();
              }
            });
          }
        }
      },
      createContext(
          /** @type {HTMLCanvasElement} */ canvas, useWebGL, setInModule,
          webGLContextAttributes) {
        if (useWebGL && Module['ctx'] && canvas == Browser.getCanvas())
          return Module['ctx'];
        // no need to recreate GL context if it's already been created for this
        // canvas.
        var ctx;
        var contextHandle;
        if (useWebGL) {
          // For GLES2/desktop GL compatibility, adjust a few defaults to be
          // different to WebGL defaults, so that they align better with the
          // desktop defaults.
          var contextAttributes = {
            antialias: false,
            alpha: false,
            majorVersion: (typeof WebGL2RenderingContext != 'undefined') ? 2 : 1
          };
          if (webGLContextAttributes) {
            for (var attribute in webGLContextAttributes) {
              contextAttributes[attribute] = webGLContextAttributes[attribute];
            }
          }
          // This check of existence of GL is here to satisfy Closure compiler,
          // which yells if variable GL is referenced below but GL object is not
          // actually compiled in because application is not doing any GL
          // operations. TODO: Ideally if GL is not being used, this function
          // Browser.createContext() should not even be emitted.
          if (typeof GL != 'undefined') {
            contextHandle = GL.createContext(canvas, contextAttributes);
            if (contextHandle) {
              ctx = GL.getContext(contextHandle).GLctx;
            }
          }
        } else {
          ctx = canvas.getContext('2d');
        }
        if (!ctx) return null;
        if (setInModule) {
          Module['ctx'] = ctx;
          if (useWebGL) GL.makeContextCurrent(contextHandle);
          Browser.useWebGL = useWebGL;
          Browser.moduleContextCreatedCallbacks.forEach(callback => callback());
          Browser.init();
        }
        return ctx;
      },
      fullscreenHandlersInstalled: false,
      lockPointer: undefined,
      resizeCanvas: undefined,
      requestFullscreen(lockPointer, resizeCanvas) {
        Browser.lockPointer = lockPointer;
        Browser.resizeCanvas = resizeCanvas;
        if (typeof Browser.lockPointer == 'undefined')
          Browser.lockPointer = true;
        if (typeof Browser.resizeCanvas == 'undefined')
          Browser.resizeCanvas = false;
        var canvas = Browser.getCanvas();
        function fullscreenChange() {
          Browser.isFullscreen = false;
          var canvasContainer = canvas.parentNode;
          if (getFullscreenElement() === canvasContainer) {
            canvas.exitFullscreen = Browser.exitFullscreen;
            if (Browser.lockPointer) canvas.requestPointerLock();
            Browser.isFullscreen = true;
            if (Browser.resizeCanvas) {
              Browser.setFullscreenCanvasSize();
            } else {
              Browser.updateCanvasDimensions(canvas);
            }
          } else {
            // remove the full screen specific parent of the canvas again to
            // restore the HTML structure from before going full screen
            canvasContainer.parentNode.insertBefore(canvas, canvasContainer);
            canvasContainer.parentNode.removeChild(canvasContainer);
            if (Browser.resizeCanvas) {
              Browser.setWindowedCanvasSize();
            } else {
              Browser.updateCanvasDimensions(canvas);
            }
          }
        }
        if (!Browser.fullscreenHandlersInstalled) {
          Browser.fullscreenHandlersInstalled = true;
          document.addEventListener('fullscreenchange', fullscreenChange);
          document.addEventListener('webkitfullscreenchange', fullscreenChange);
        }
        // create a new parent to ensure the canvas has no siblings. this allows
        // browsers to optimize full screen performance when its parent is the
        // full screen root
        var canvasContainer = document.createElement('div');
        canvas.parentNode.insertBefore(canvasContainer, canvas);
        canvasContainer.appendChild(canvas);
        // use parent of canvas as full screen root to allow aspect ratio
        // correction (Firefox stretches the root to screen size) Safari didn't
        // support Element.requestFullscreen until 16.4 See:
        // https://developer.mozilla.org/en-US/docs/Web/API/Element/requestFullscreen
        /** @suppress {checkTypes} */ canvasContainer.requestFullscreen ??=
            (canvasContainer['webkitRequestFullscreen'] ?
                 () => canvasContainer['webkitRequestFullscreen'](
                     Element.ALLOW_KEYBOARD_INPUT) :
                 null) ??
            (canvasContainer['webkitRequestFullScreen'] ?
                 () => canvasContainer['webkitRequestFullScreen'](
                     Element.ALLOW_KEYBOARD_INPUT) :
                 null);
        canvasContainer.requestFullscreen();
      },
      exitFullscreen() {
        // This is workaround for chrome. Trying to exit from fullscreen
        // not in fullscreen state will cause 'TypeError: Document not active'
        // in chrome. See
        // https://github.com/emscripten-core/emscripten/pull/8236
        if (!Browser.isFullscreen) {
          return false;
        }
        var CFS = document.exitFullscreen ?? document['webkitCancelFullScreen'];
        CFS.apply(document, []);
        return true;
      },
      safeSetTimeout(func, timeout) {
        // Legacy function, this is used by the SDL2 port so we need to keep it
        // around at least until that is updated.
        // See https://github.com/libsdl-org/SDL/pull/6304
        return safeSetTimeout(func, timeout);
      },
      getMimetype(name) {
        return {
          'jpg': 'image/jpeg',
          'jpeg': 'image/jpeg',
          'png': 'image/png',
          'bmp': 'image/bmp',
          'ogg': 'audio/ogg',
          'wav': 'audio/wav',
          'mp3': 'audio/mpeg'
        }[name.slice(name.lastIndexOf('.') + 1)];
      },
      getUserMedia(func) {
        return navigator.mediaDevices.getUserMedia(func);
      },
      getMouseWheelDelta(event) {
        var delta = 0;
        switch (event.type) {
          case 'DOMMouseScroll':
            // 3 lines make up a step
            delta = event.detail / 3;
            break;

          case 'mousewheel':
            // 120 units make up a step
            delta = event.wheelDelta / 120;
            break;

          case 'wheel':
            delta = event.deltaY;
            switch (event.deltaMode) {
              case 0:
                // DOM_DELTA_PIXEL: 100 pixels make up a step
                delta /= 100;
                break;

              case 1:
                // DOM_DELTA_LINE: 3 lines make up a step
                delta /= 3;
                break;

              case 2:
                // DOM_DELTA_PAGE: A page makes up 80 steps
                delta *= 80;
                break;

              default:
                abort(
                    'unrecognized mouse wheel delta mode: ' + event.deltaMode);
            }
            break;

          default:
            abort('unrecognized mouse wheel event: ' + event.type);
        }
        return delta;
      },
      mouseX: 0,
      mouseY: 0,
      mouseMovementX: 0,
      mouseMovementY: 0,
      touches: {},
      lastTouches: {},
      calculateMouseCoords(pageX, pageY) {
        // Calculate the movement based on the changes
        // in the coordinates.
        var canvas = Browser.getCanvas();
        var rect = canvas.getBoundingClientRect();
        var adjustedX = pageX - (window.scrollX + rect.left);
        var adjustedY = pageY - (window.scrollY + rect.top);
        // the canvas might be CSS-scaled compared to its backbuffer;
        // SDL-using content will want mouse coordinates in terms
        // of backbuffer units.
        adjustedX = adjustedX * (canvas.width / rect.width);
        adjustedY = adjustedY * (canvas.height / rect.height);
        return {x: adjustedX, y: adjustedY};
      },
      setMouseCoords(pageX, pageY) {
        const {x, y} = Browser.calculateMouseCoords(pageX, pageY);
        Browser.mouseMovementX = x - Browser.mouseX;
        Browser.mouseMovementY = y - Browser.mouseY;
        Browser.mouseX = x;
        Browser.mouseY = y;
      },
      calculateMouseEvent(event) {
        // event should be mousemove, mousedown or mouseup
        if (Browser.pointerLock) {
          // When the pointer is locked, calculate the coordinates
          // based on the movement of the mouse.
          Browser.mouseMovementX = event.movementX;
          Browser.mouseMovementY = event.movementY;
          // add the mouse delta to the current absolute mouse position
          Browser.mouseX += Browser.mouseMovementX;
          Browser.mouseY += Browser.mouseMovementY;
        } else {
          if (event.type === 'touchstart' || event.type === 'touchend' ||
              event.type === 'touchmove') {
            var touch = event.touch;
            if (touch === undefined) {
              return;
            }
            var coords = Browser.calculateMouseCoords(touch.pageX, touch.pageY);
            if (event.type === 'touchstart') {
              Browser.lastTouches[touch.identifier] = coords;
              Browser.touches[touch.identifier] = coords;
            } else if (
                event.type === 'touchend' || event.type === 'touchmove') {
              var last = Browser.touches[touch.identifier];
              last ||= coords;
              Browser.lastTouches[touch.identifier] = last;
              Browser.touches[touch.identifier] = coords;
            }
            return;
          }
          Browser.setMouseCoords(event.pageX, event.pageY);
        }
      },
      resizeListeners: [],
      updateResizeListeners() {
        var canvas = Browser.getCanvas();
        Browser.resizeListeners.forEach(
            listener => listener(canvas.width, canvas.height));
      },
      setCanvasSize(width, height, noUpdates) {
        var canvas = Browser.getCanvas();
        Browser.updateCanvasDimensions(canvas, width, height);
        if (!noUpdates) Browser.updateResizeListeners();
      },
      windowedWidth: 0,
      windowedHeight: 0,
      setFullscreenCanvasSize() {
        // check if SDL is available
        if (typeof SDL != 'undefined') {
          var flags = HEAPU32[((SDL.screen) >>> 2) >>> 0];
          flags = flags | 8388608;
          // set SDL_FULLSCREEN flag
          HEAP32[((SDL.screen) >>> 2) >>> 0] = flags;
        }
        Browser.updateCanvasDimensions(Browser.getCanvas());
        Browser.updateResizeListeners();
      },
      setWindowedCanvasSize() {
        // check if SDL is available
        if (typeof SDL != 'undefined') {
          var flags = HEAPU32[((SDL.screen) >>> 2) >>> 0];
          flags = flags & ~8388608;
          // clear SDL_FULLSCREEN flag
          HEAP32[((SDL.screen) >>> 2) >>> 0] = flags;
        }
        Browser.updateCanvasDimensions(Browser.getCanvas());
        Browser.updateResizeListeners();
      },
      updateCanvasDimensions(canvas, wNative, hNative) {
        if (wNative && hNative) {
          canvas.widthNative = wNative;
          canvas.heightNative = hNative;
        } else {
          wNative = canvas.widthNative;
          hNative = canvas.heightNative;
        }
        var w = wNative;
        var h = hNative;
        if ((getFullscreenElement() === canvas.parentNode) &&
            (typeof screen != 'undefined')) {
          var factor = Math.min(screen.width / w, screen.height / h);
          w = Math.round(w * factor);
          h = Math.round(h * factor);
        }
        if (Browser.resizeCanvas) {
          if (canvas.width != w) canvas.width = w;
          if (canvas.height != h) canvas.height = h;
          if (typeof canvas.style != 'undefined') {
            canvas.style.removeProperty('width');
            canvas.style.removeProperty('height');
          }
        } else {
          if (canvas.width != wNative) canvas.width = wNative;
          if (canvas.height != hNative) canvas.height = hNative;
          if (typeof canvas.style != 'undefined') {
            if (w != wNative || h != hNative) {
              canvas.style.setProperty('width', w + 'px', 'important');
              canvas.style.setProperty('height', h + 'px', 'important');
            } else {
              canvas.style.removeProperty('width');
              canvas.style.removeProperty('height');
            }
          }
        }
      }
    };

    /** @type {!Int8Array} */ var HEAP8;

    var callRuntimeCallbacks = callbacks => {
      while (callbacks.length > 0) {
        // Pass the module as the first argument.
        callbacks.shift()(Module);
      }
    };

    var onPostRuns = [];

    var onPreRuns = [];

    var dynCalls = {};

    var dynCallLegacy = (sig, ptr, args) => {
      sig = sig.replace(/p/g, 'i');
      var f = dynCalls[sig];
      return f(ptr, ...args);
    };

    var dynCall = (sig, ptr, args = [], promising = false) => {
      var rtn = dynCallLegacy(sig, ptr, args);
      function convert(rtn) {
        return sig[0] == 'p' ? rtn >>> 0 : rtn;
      }
      return convert(rtn);
    };

    var noExitRuntime = true;

    var stackRestore = val => __emscripten_stack_restore(val);

    var stackSave = () => _emscripten_stack_get_current();

    class ExceptionInfo {
      // excPtr - Thrown object pointer to wrap. Metadata pointer is calculated
      // from it.
      constructor(excPtr) {
        this.excPtr = excPtr;
        this.ptr = excPtr - 24;
      }
      set_type(type) {
        HEAPU32[(((this.ptr) + (4)) >>> 2) >>> 0] = type;
      }
      get_type() {
        return HEAPU32[(((this.ptr) + (4)) >>> 2) >>> 0];
      }
      set_destructor(destructor) {
        HEAPU32[(((this.ptr) + (8)) >>> 2) >>> 0] = destructor;
      }
      get_destructor() {
        return HEAPU32[(((this.ptr) + (8)) >>> 2) >>> 0];
      }
      set_caught(caught) {
        caught = caught ? 1 : 0;
        HEAP8[(this.ptr) + (12) >>> 0] = caught;
      }
      get_caught() {
        return HEAP8[(this.ptr) + (12) >>> 0] != 0;
      }
      set_rethrown(rethrown) {
        rethrown = rethrown ? 1 : 0;
        HEAP8[(this.ptr) + (13) >>> 0] = rethrown;
      }
      get_rethrown() {
        return HEAP8[(this.ptr) + (13) >>> 0] != 0;
      }
      // Initialize native structure fields. Should be called once after
      // allocated.
      init(type, destructor) {
        this.set_adjusted_ptr(0);
        this.set_type(type);
        this.set_destructor(destructor);
      }
      set_adjusted_ptr(adjustedPtr) {
        HEAPU32[(((this.ptr) + (16)) >>> 2) >>> 0] = adjustedPtr;
      }
      get_adjusted_ptr() {
        return HEAPU32[(((this.ptr) + (16)) >>> 2) >>> 0];
      }
    }

    var uncaughtExceptionCount = 0;

    var convertI32PairToI53Checked = (lo, hi) =>
        ((hi + 2097152) >>> 0 < 4194305 - !!lo) ? (lo >>> 0) + hi * 4294967296 :
                                                  NaN;

    function ___cxa_throw(ptr, type, destructor) {
      ptr >>>= 0;
      type >>>= 0;
      destructor >>>= 0;
      var info = new ExceptionInfo(ptr);
      // Initialize ExceptionInfo content after it was allocated in
      // __cxa_allocate_exception.
      info.init(type, destructor);
      uncaughtExceptionCount++;
      abort();
    }

    function __Unwind_RaiseException(ex) {
      ex >>>= 0;
      err('Warning: _Unwind_RaiseException is not correctly implemented');
      return ___cxa_throw(ex, 0, 0);
    }

    var UTF8Decoder = new TextDecoder;

    /**
     * heapOrArray is either a regular array, or a JavaScript typed array view.
     * @param {number} idx
     * @param {number=} maxBytesToRead
     * @param {boolean=} ignoreNul
     * @return {number}
     */
    var findStringEnd = (heapOrArray, idx, maxBytesToRead, ignoreNul) => {
      var maxIdx = idx + maxBytesToRead;
      if (ignoreNul) return maxIdx;
      // TextDecoder needs to know the byte length in advance, it doesn't stop
      // on null terminator by itself. As a tiny code save trick, compare idx
      // against maxIdx using a negation, so that maxBytesToRead=undefined/NaN
      // means Infinity.
      while (heapOrArray[idx] && !(idx >= maxIdx)) ++idx;
      return idx;
    };

    /** @type {!Uint8Array} */ var HEAPU8;

    /**
     * Given a pointer 'ptr' to a null-terminated UTF8-encoded string in the
     * emscripten HEAP, returns a copy of that string as a Javascript String
     * object.
     *
     * @param {number} ptr
     * @param {number=} maxBytesToRead - An optional length that specifies the
     *   maximum number of bytes to read. You can omit this parameter to scan
     * the string until the first 0 byte. If maxBytesToRead is passed, and the
     * string at [ptr, ptr+maxBytesToReadr[ contains a null byte in the middle,
     * then the string will cut short at that byte index.
     * @param {boolean=} ignoreNul - If true, the function will not stop on a
     *     NUL character.
     * @return {string}
     */
    var UTF8ToString = (ptr, maxBytesToRead, ignoreNul) => {
      ptr >>>= 0;
      if (!ptr) return '';
      var end = findStringEnd(HEAPU8, ptr, maxBytesToRead, ignoreNul);
      return UTF8Decoder.decode(HEAPU8.subarray(ptr >>> 0, end >>> 0));
    };

    function ___assert_fail(condition, filename, line, func) {
      condition >>>= 0;
      filename >>>= 0;
      func >>>= 0;
      return abort(`Assertion failed: ${UTF8ToString(condition)}, at: ` + [
        filename ? UTF8ToString(filename) : 'unknown filename', line,
        func ? UTF8ToString(func) : 'unknown function'
      ]);
    }

    var PATH = {
      isAbs: path => path.charAt(0) === '/',
      splitPath: filename => {
        var splitPathRe =
            /^(\/?|)([\s\S]*?)((?:\.{1,2}|[^\/]+?|)(\.[^.\/]*|))(?:[\/]*)$/;
        return splitPathRe.exec(filename).slice(1);
      },
      normalizeArray: (parts, allowAboveRoot) => {
        // if the path tries to go above the root, `up` ends up > 0
        var up = 0;
        for (var i = parts.length - 1; i >= 0; i--) {
          var last = parts[i];
          if (last === '.') {
            parts.splice(i, 1);
          } else if (last === '..') {
            parts.splice(i, 1);
            up++;
          } else if (up) {
            parts.splice(i, 1);
            up--;
          }
        }
        // if the path is allowed to go above the root, restore leading ..s
        if (allowAboveRoot) {
          for (; up; up--) {
            parts.unshift('..');
          }
        }
        return parts;
      },
      normalize: path => {
        var isAbsolute = PATH.isAbs(path),
            trailingSlash = path.slice(-1) === '/';
        // Normalize the path
        path =
            PATH.normalizeArray(path.split('/').filter(p => !!p), !isAbsolute)
                .join('/');
        if (!path && !isAbsolute) {
          path = '.';
        }
        if (path && trailingSlash) {
          path += '/';
        }
        return (isAbsolute ? '/' : '') + path;
      },
      dirname: path => {
        var result = PATH.splitPath(path), root = result[0], dir = result[1];
        if (!root && !dir) {
          // No dirname whatsoever
          return '.';
        }
        if (dir) {
          // It has a dirname, strip trailing slash
          dir = dir.slice(0, -1);
        }
        return root + dir;
      },
      basename: path => path && path.match(/([^\/]+|\/)\/*$/)[1],
      join: (...paths) => PATH.normalize(paths.join('/')),
      join2: (l, r) => PATH.normalize(l + '/' + r)
    };

    var initRandomFill = () => {
      // This block is not needed on v19+ since crypto.getRandomValues is
      // builtin
      if (ENVIRONMENT_IS_NODE) {
        var nodeCrypto = require('node:crypto');
        return view => (nodeCrypto.randomFillSync(view), 0);
      }
      return view => (crypto.getRandomValues(view), 0);
    };

    var randomFill = view => (randomFill = initRandomFill())(view);

    var PATH_FS = {
      resolve: (...args) => {
        var resolvedPath = '', resolvedAbsolute = false;
        for (var i = args.length - 1; i >= -1 && !resolvedAbsolute; i--) {
          var path = (i >= 0) ? args[i] : FS.cwd();
          // Skip empty and invalid entries
          if (typeof path != 'string') {
            throw new TypeError('Arguments to path.resolve must be strings');
          } else if (!path) {
            return '';
          }
          resolvedPath = path + '/' + resolvedPath;
          resolvedAbsolute = PATH.isAbs(path);
        }
        // At this point the path should be resolved to a full absolute path,
        // but handle relative paths to be safe (might happen when process.cwd()
        // fails)
        resolvedPath =
            PATH.normalizeArray(
                    resolvedPath.split('/').filter(p => !!p), !resolvedAbsolute)
                .join('/');
        return ((resolvedAbsolute ? '/' : '') + resolvedPath) || '.';
      },
      relative: (from, to) => {
        from = PATH_FS.resolve(from).slice(1);
        to = PATH_FS.resolve(to).slice(1);
        function trim(arr) {
          var start = 0;
          for (; start < arr.length; start++) {
            if (arr[start] !== '') break;
          }
          var end = arr.length - 1;
          for (; end >= 0; end--) {
            if (arr[end] !== '') break;
          }
          if (start > end) return [];
          return arr.slice(start, end - start + 1);
        }
        var fromParts = trim(from.split('/'));
        var toParts = trim(to.split('/'));
        var length = Math.min(fromParts.length, toParts.length);
        var samePartsLength = length;
        for (var i = 0; i < length; i++) {
          if (fromParts[i] !== toParts[i]) {
            samePartsLength = i;
            break;
          }
        }
        var outputParts = [];
        for (var i = samePartsLength; i < fromParts.length; i++) {
          outputParts.push('..');
        }
        outputParts = outputParts.concat(toParts.slice(samePartsLength));
        return outputParts.join('/');
      }
    };

    /**
     * Given a pointer 'idx' to a null-terminated UTF8-encoded string in the
     * given array that contains uint8 values, returns a copy of that string as
     * a Javascript String object. heapOrArray is either a regular array, or a
     * JavaScript typed array view.
     * @param {number=} idx
     * @param {number=} maxBytesToRead
     * @param {boolean=} ignoreNul - If true, the function will not stop on a
     *     NUL character.
     * @return {string}
     */
    var UTF8ArrayToString =
        (heapOrArray, idx = 0, maxBytesToRead, ignoreNul) => {
          idx >>>= 0;
          var endPtr =
              findStringEnd(heapOrArray, idx, maxBytesToRead, ignoreNul);
          return UTF8Decoder.decode(
              heapOrArray.buffer ?
                  heapOrArray.subarray(idx, endPtr) :
                  new Uint8Array(heapOrArray.slice(idx, endPtr)));
        };

    var FS_stdin_getChar_buffer = [];

    var lengthBytesUTF8 = str => {
      var len = 0;
      for (var i = 0; i < str.length; ++i) {
        // Gotcha: charCodeAt returns a 16-bit word that is a UTF-16 encoded
        // code unit, not a Unicode code point of the character! So decode
        // UTF16->UTF32->UTF8.
        // See http://unicode.org/faq/utf_bom.html#utf16-3
        var c = str.charCodeAt(i);
        // possibly a lead surrogate
        if (c <= 127) {
          len++;
        } else if (c <= 2047) {
          len += 2;
        } else if (c >= 55296 && c <= 57343) {
          len += 4;
          ++i;
        } else {
          len += 3;
        }
      }
      return len;
    };

    var stringToUTF8Array = (str, heap, outIdx, maxBytesToWrite) => {
      outIdx >>>= 0;
      // Parameter maxBytesToWrite is not optional. Negative values, 0, null,
      // undefined and false each don't write out any bytes.
      if (!(maxBytesToWrite > 0)) return 0;
      var startIdx = outIdx;
      var endIdx = outIdx + maxBytesToWrite - 1;
      // -1 for string null terminator.
      for (var i = 0; i < str.length; ++i) {
        // For UTF8 byte structure, see
        // http://en.wikipedia.org/wiki/UTF-8#Description and
        // https://www.ietf.org/rfc/rfc2279.txt and
        // https://tools.ietf.org/html/rfc3629
        var u = str.codePointAt(i);
        if (u <= 127) {
          if (outIdx >= endIdx) break;
          heap[outIdx++ >>> 0] = u;
        } else if (u <= 2047) {
          if (outIdx + 1 >= endIdx) break;
          heap[outIdx++ >>> 0] = 192 | (u >> 6);
          heap[outIdx++ >>> 0] = 128 | (u & 63);
        } else if (u <= 65535) {
          if (outIdx + 2 >= endIdx) break;
          heap[outIdx++ >>> 0] = 224 | (u >> 12);
          heap[outIdx++ >>> 0] = 128 | ((u >> 6) & 63);
          heap[outIdx++ >>> 0] = 128 | (u & 63);
        } else {
          if (outIdx + 3 >= endIdx) break;
          heap[outIdx++ >>> 0] = 240 | (u >> 18);
          heap[outIdx++ >>> 0] = 128 | ((u >> 12) & 63);
          heap[outIdx++ >>> 0] = 128 | ((u >> 6) & 63);
          heap[outIdx++ >>> 0] = 128 | (u & 63);
          // Gotcha: if codePoint is over 0xFFFF, it is represented as a
          // surrogate pair in UTF-16. We need to manually skip over the second
          // code unit for correct iteration.
          i++;
        }
      }
      // Null-terminate the pointer to the buffer.
      heap[outIdx >>> 0] = 0;
      return outIdx - startIdx;
    };

    /** @type {function(string, boolean=, number=)} */ var intArrayFromString =
        (stringy, dontAddNull, length) => {
          var len = length > 0 ? length : lengthBytesUTF8(stringy) + 1;
          var u8array = new Array(len);
          var numBytesWritten =
              stringToUTF8Array(stringy, u8array, 0, u8array.length);
          if (dontAddNull) u8array.length = numBytesWritten;
          return u8array;
        };

    var FS_stdin_getChar = () => {
      if (!FS_stdin_getChar_buffer.length) {
        var result = null;
        if (ENVIRONMENT_IS_NODE) {
          // we will read data by chunks of BUFSIZE
          var BUFSIZE = 256;
          var buf = Buffer.alloc(BUFSIZE);
          var bytesRead = 0;
          // For some reason we must suppress a closure warning here, even
          // though fd definitely exists on process.stdin, and is even the
          // proper way to get the fd of stdin,
          // https://github.com/nodejs/help/issues/2136#issuecomment-523649904
          // This started to happen after moving this logic out of
          // library_tty.js, so it is related to the surrounding code in some
          // unclear manner.
          /** @suppress {missingProperties} */ var fd = process.stdin.fd;
          try {
            bytesRead = fs.readSync(fd, buf, 0, BUFSIZE);
          } catch (e) {
            // Cross-platform differences: on Windows, reading EOF throws an
            // exception, but on other OSes, reading EOF returns 0. Uniformize
            // behavior by treating the EOF exception to return 0.
            if (e.toString().includes('EOF'))
              bytesRead = 0;
            else
              throw e;
          }
          if (bytesRead > 0) {
            result = buf.slice(0, bytesRead).toString('utf-8');
          }
        } else if (globalThis.window?.prompt) {
          // Browser.
          result = window.prompt('Input: ');
          // returns null on cancel
          if (result !== null) {
            result += '\n';
          }
        } else {
        }
        if (!result) {
          return null;
        }
        FS_stdin_getChar_buffer = intArrayFromString(result, true);
      }
      return FS_stdin_getChar_buffer.shift();
    };

    var TTY = {
      ttys: [],
      init() {},
      shutdown() {},
      register(dev, ops) {
        TTY.ttys[dev] = {input: [], output: [], ops};
        FS.registerDevice(dev, TTY.stream_ops);
      },
      stream_ops: {
        open(stream) {
          var tty = TTY.ttys[stream.node.rdev];
          if (!tty) {
            throw new FS.ErrnoError(43);
          }
          stream.tty = tty;
          stream.seekable = false;
        },
        close(stream) {
          // flush any pending line data
          stream.tty.ops.fsync(stream.tty);
        },
        fsync(stream) {
          stream.tty.ops.fsync(stream.tty);
        },
        read(stream, buffer, offset, length, pos) {
          if (!stream.tty || !stream.tty.ops.get_char) {
            throw new FS.ErrnoError(60);
          }
          var bytesRead = 0;
          for (var i = 0; i < length; i++) {
            var result;
            try {
              result = stream.tty.ops.get_char(stream.tty);
            } catch (e) {
              throw new FS.ErrnoError(29);
            }
            if (result === undefined && bytesRead === 0) {
              throw new FS.ErrnoError(6);
            }
            if (result === null || result === undefined) break;
            bytesRead++;
            buffer[offset + i] = result;
          }
          if (bytesRead) {
            stream.node.atime = Date.now();
          }
          return bytesRead;
        },
        write(stream, buffer, offset, length, pos) {
          if (!stream.tty || !stream.tty.ops.put_char) {
            throw new FS.ErrnoError(60);
          }
          try {
            for (var i = 0; i < length; i++) {
              stream.tty.ops.put_char(stream.tty, buffer[offset + i]);
            }
          } catch (e) {
            throw new FS.ErrnoError(29);
          }
          if (length) {
            stream.node.mtime = stream.node.ctime = Date.now();
          }
          return i;
        }
      },
      default_tty_ops: {
        get_char(tty) {
          return FS_stdin_getChar();
        },
        put_char(tty, val) {
          if (val === null || val === 10) {
            out(UTF8ArrayToString(tty.output));
            tty.output = [];
          } else {
            if (val != 0) tty.output.push(val);
          }
        },
        fsync(tty) {
          if (tty.output?.length > 0) {
            out(UTF8ArrayToString(tty.output));
            tty.output = [];
          }
        },
        ioctl_tcgets(tty) {
          // typical setting
          return {
            c_iflag: 25856,
            c_oflag: 5,
            c_cflag: 191,
            c_lflag: 35387,
            c_cc: [
              3, 28, 127, 21, 4, 0, 1, 0, 17, 19, 26, 0, 18, 15, 23, 22,
              0, 0,  0,   0,  0, 0, 0, 0, 0,  0,  0,  0, 0,  0,  0,  0
            ]
          };
        },
        ioctl_tcsets(tty, optional_actions, data) {
          // currently just ignore
          return 0;
        },
        ioctl_tiocgwinsz(tty) {
          return [24, 80];
        }
      },
      default_tty1_ops: {
        put_char(tty, val) {
          if (val === null || val === 10) {
            err(UTF8ArrayToString(tty.output));
            tty.output = [];
          } else {
            if (val != 0) tty.output.push(val);
          }
        },
        fsync(tty) {
          if (tty.output?.length > 0) {
            err(UTF8ArrayToString(tty.output));
            tty.output = [];
          }
        }
      }
    };

    var zeroMemory = (ptr, size) => HEAPU8.fill(0, ptr, ptr + size);

    var alignMemory = (size, alignment) =>
        Math.ceil(size / alignment) * alignment;

    var mmapAlloc = size => {
      size = alignMemory(size, 65536);
      var ptr = _emscripten_builtin_memalign(65536, size);
      if (ptr) zeroMemory(ptr, size);
      return ptr;
    };

    var MEMFS = {
      ops_table: null,
      mount(mount) {
        return MEMFS.createNode(null, '/', 16895, 0);
      },
      createNode(parent, name, mode, dev) {
        if (FS.isBlkdev(mode) || FS.isFIFO(mode)) {
          // not supported
          throw new FS.ErrnoError(63);
        }
        MEMFS.ops_table ||= {
          dir: {
            node: {
              getattr: MEMFS.node_ops.getattr,
              setattr: MEMFS.node_ops.setattr,
              lookup: MEMFS.node_ops.lookup,
              mknod: MEMFS.node_ops.mknod,
              rename: MEMFS.node_ops.rename,
              unlink: MEMFS.node_ops.unlink,
              rmdir: MEMFS.node_ops.rmdir,
              readdir: MEMFS.node_ops.readdir,
              symlink: MEMFS.node_ops.symlink
            },
            stream: {llseek: MEMFS.stream_ops.llseek}
          },
          file: {
            node: {
              getattr: MEMFS.node_ops.getattr,
              setattr: MEMFS.node_ops.setattr
            },
            stream: {
              llseek: MEMFS.stream_ops.llseek,
              read: MEMFS.stream_ops.read,
              write: MEMFS.stream_ops.write,
              mmap: MEMFS.stream_ops.mmap,
              msync: MEMFS.stream_ops.msync
            }
          },
          link: {
            node: {
              getattr: MEMFS.node_ops.getattr,
              setattr: MEMFS.node_ops.setattr,
              readlink: MEMFS.node_ops.readlink
            },
            stream: {}
          },
          chrdev: {
            node: {
              getattr: MEMFS.node_ops.getattr,
              setattr: MEMFS.node_ops.setattr
            },
            stream: FS.chrdev_stream_ops
          }
        };
        var node = FS.createNode(parent, name, mode, dev);
        if (FS.isDir(node.mode)) {
          node.node_ops = MEMFS.ops_table.dir.node;
          node.stream_ops = MEMFS.ops_table.dir.stream;
          node.contents = {};
        } else if (FS.isFile(node.mode)) {
          node.node_ops = MEMFS.ops_table.file.node;
          node.stream_ops = MEMFS.ops_table.file.stream;
          // The actual number of bytes used in the typed array, as opposed to
          // contents.length which gives the whole capacity.
          node.usedBytes = 0;
          // The byte data of the file is stored in a typed array.
          // Note: typed arrays are not resizable like normal JS arrays are, so
          // there is a small penalty involved for appending file writes that
          // continuously grow a file similar to std::vector capacity vs used.
          node.contents = MEMFS.emptyFileContents ??= new Uint8Array(0);
        } else if (FS.isLink(node.mode)) {
          node.node_ops = MEMFS.ops_table.link.node;
          node.stream_ops = MEMFS.ops_table.link.stream;
        } else if (FS.isChrdev(node.mode)) {
          node.node_ops = MEMFS.ops_table.chrdev.node;
          node.stream_ops = MEMFS.ops_table.chrdev.stream;
        }
        node.atime = node.mtime = node.ctime = Date.now();
        // add the new node to the parent
        if (parent) {
          parent.contents[name] = node;
          parent.atime = parent.mtime = parent.ctime = node.atime;
        }
        return node;
      },
      getFileDataAsTypedArray(node) {
        return node.contents.subarray(0, node.usedBytes);
      },
      expandFileStorage(node, newCapacity) {
        var prevCapacity = node.contents.length;
        if (prevCapacity >= newCapacity) return;
        // No need to expand, the storage was already large enough.
        // Don't expand strictly to the given requested limit if it's only a
        // very small increase, but instead geometrically grow capacity. For
        // small filesizes (<1MB), perform size*2 geometric increase, but for
        // large sizes, do a much more conservative size*1.125 increase to avoid
        // overshooting the allocation cap by a very large margin.
        var CAPACITY_DOUBLING_MAX = 1024 * 1024;
        newCapacity = Math.max(
            newCapacity,
            (prevCapacity *
             (prevCapacity < CAPACITY_DOUBLING_MAX ? 2 : 1.125)) >>>
                0);
        if (prevCapacity) newCapacity = Math.max(newCapacity, 256);
        // At minimum allocate 256b for each file when expanding.
        var oldContents = MEMFS.getFileDataAsTypedArray(node);
        node.contents = new Uint8Array(newCapacity);
        // Allocate new storage.
        node.contents.set(oldContents);
      },
      resizeFileStorage(node, newSize) {
        if (node.usedBytes == newSize) return;
        var oldContents = node.contents;
        node.contents = new Uint8Array(newSize);
        // Allocate new storage.
        node.contents.set(
            oldContents.subarray(0, Math.min(newSize, node.usedBytes)));
        // Copy old data over to the new storage.
        node.usedBytes = newSize;
      },
      node_ops: {
        getattr(node) {
          var attr = {};
          // device numbers reuse inode numbers.
          attr.dev = FS.isChrdev(node.mode) ? node.id : 1;
          attr.ino = node.id;
          attr.mode = node.mode;
          attr.nlink = 1;
          attr.uid = 0;
          attr.gid = 0;
          attr.rdev = node.rdev;
          if (FS.isDir(node.mode)) {
            attr.size = 4096;
          } else if (FS.isFile(node.mode)) {
            attr.size = node.usedBytes;
          } else if (FS.isLink(node.mode)) {
            attr.size = node.link.length;
          } else {
            attr.size = 0;
          }
          attr.atime = new Date(node.atime);
          attr.mtime = new Date(node.mtime);
          attr.ctime = new Date(node.ctime);
          // NOTE: In our implementation, st_blocks =
          // Math.ceil(st_size/st_blksize),
          //       but this is not required by the standard.
          attr.blksize = 4096;
          attr.blocks = Math.ceil(attr.size / attr.blksize);
          return attr;
        },
        setattr(node, attr) {
          for (const key of ['mode', 'atime', 'mtime', 'ctime']) {
            if (attr[key] != null) {
              node[key] = attr[key];
            }
          }
          if (attr.size !== undefined) {
            MEMFS.resizeFileStorage(node, attr.size);
          }
        },
        lookup(parent, name) {
          // This error may happen quite a bit. To avoid overhead we reuse it
          // (and suffer a lack of stack info).
          if (!MEMFS.doesNotExistError) {
            MEMFS.doesNotExistError = new FS.ErrnoError(44);
            /** @suppress {checkTypes} */ MEMFS.doesNotExistError.stack =
                '<generic error, no stack>';
          }
          throw MEMFS.doesNotExistError;
        },
        mknod(parent, name, mode, dev) {
          return MEMFS.createNode(parent, name, mode, dev);
        },
        rename(old_node, new_dir, new_name) {
          var new_node;
          try {
            new_node = FS.lookupNode(new_dir, new_name);
          } catch (e) {
          }
          if (new_node) {
            if (FS.isDir(old_node.mode)) {
              // if we're overwriting a directory at new_name, make sure it's
              // empty.
              for (var i in new_node.contents) {
                throw new FS.ErrnoError(55);
              }
            }
            FS.hashRemoveNode(new_node);
          }
          // do the internal rewiring
          delete old_node.parent.contents[old_node.name];
          new_dir.contents[new_name] = old_node;
          old_node.name = new_name;
          new_dir.ctime = new_dir.mtime = old_node.parent.ctime =
              old_node.parent.mtime = Date.now();
        },
        unlink(parent, name) {
          delete parent.contents[name];
          parent.ctime = parent.mtime = Date.now();
        },
        rmdir(parent, name) {
          var node = FS.lookupNode(parent, name);
          for (var i in node.contents) {
            throw new FS.ErrnoError(55);
          }
          delete parent.contents[name];
          parent.ctime = parent.mtime = Date.now();
        },
        readdir(node) {
          return ['.', '..', ...Object.keys(node.contents)];
        },
        symlink(parent, newname, oldpath) {
          var node = MEMFS.createNode(parent, newname, 511 | 40960, 0);
          node.link = oldpath;
          return node;
        },
        readlink(node) {
          if (!FS.isLink(node.mode)) {
            throw new FS.ErrnoError(28);
          }
          return node.link;
        }
      },
      stream_ops: {
        read(stream, buffer, offset, length, position) {
          var contents = stream.node.contents;
          if (position >= stream.node.usedBytes) return 0;
          var size = Math.min(stream.node.usedBytes - position, length);
          buffer.set(contents.subarray(position, position + size), offset);
          return size;
        },
        write(stream, buffer, offset, length, position, canOwn) {
          // If the buffer is located in main memory (HEAP), and if
          // memory can grow, we can't hold on to references of the
          // memory buffer, as they may get invalidated. That means we
          // need to copy its contents.
          if (buffer.buffer === HEAP8.buffer) {
            canOwn = false;
          }
          if (!length) return 0;
          var node = stream.node;
          node.mtime = node.ctime = Date.now();
          if (canOwn) {
            node.contents = buffer.subarray(offset, offset + length);
            node.usedBytes = length;
          } else if (node.usedBytes === 0 && position === 0) {
            // If this is a simple first write to an empty file, do a fast set
            // since we don't need to care about old data.
            node.contents = buffer.slice(offset, offset + length);
            node.usedBytes = length;
          } else {
            MEMFS.expandFileStorage(node, position + length);
            // Use typed array write which is available.
            node.contents.set(
                buffer.subarray(offset, offset + length), position);
            node.usedBytes = Math.max(node.usedBytes, position + length);
          }
          return length;
        },
        llseek(stream, offset, whence) {
          var position = offset;
          if (whence === 1) {
            position += stream.position;
          } else if (whence === 2) {
            if (FS.isFile(stream.node.mode)) {
              position += stream.node.usedBytes;
            }
          }
          if (position < 0) {
            throw new FS.ErrnoError(28);
          }
          return position;
        },
        mmap(stream, length, position, prot, flags) {
          if (!FS.isFile(stream.node.mode)) {
            throw new FS.ErrnoError(43);
          }
          var ptr;
          var allocated;
          var contents = stream.node.contents;
          // Only make a new copy when MAP_PRIVATE is specified.
          if (!(flags & 2) && contents.buffer === HEAP8.buffer) {
            // We can't emulate MAP_SHARED when the file is not backed by the
            // buffer we're mapping to (e.g. the HEAP buffer).
            allocated = false;
            ptr = contents.byteOffset;
          } else {
            allocated = true;
            ptr = mmapAlloc(length);
            if (!ptr) {
              throw new FS.ErrnoError(48);
            }
            if (contents) {
              // Try to avoid unnecessary slices.
              if (position > 0 || position + length < contents.length) {
                if (contents.subarray) {
                  contents = contents.subarray(position, position + length);
                } else {
                  contents = Array.prototype.slice.call(
                      contents, position, position + length);
                }
              }
              HEAP8.set(contents, ptr >>> 0);
            }
          }
          return {ptr, allocated};
        },
        msync(stream, buffer, offset, length, mmapFlags) {
          MEMFS.stream_ops.write(stream, buffer, 0, length, offset, false);
          // should we check if bytesWritten and length are the same?
          return 0;
        }
      }
    };

    var FS_modeStringToFlags = str => {
      if (typeof str != 'string') return str;
      var flagModes = {
        'r': 0,
        'r+': 2,
        'w': 512 | 64 | 1,
        'w+': 512 | 64 | 2,
        'a': 1024 | 64 | 1,
        'a+': 1024 | 64 | 2
      };
      var flags = flagModes[str];
      if (typeof flags == 'undefined') {
        throw new Error(`Unknown file open mode: ${str}`);
      }
      return flags;
    };

    var FS_fileDataToTypedArray = data => {
      if (typeof data == 'string') {
        data = intArrayFromString(data, true);
      }
      if (!data.subarray) {
        data = new Uint8Array(data);
      }
      return data;
    };

    var FS_getMode = (canRead, canWrite) => {
      var mode = 0;
      if (canRead) mode |= 292 | 73;
      if (canWrite) mode |= 146;
      return mode;
    };

    var asyncLoad = async url => {
      var arrayBuffer = await readAsync(url);
      return new Uint8Array(arrayBuffer);
    };

    var FS_createDataFile = (...args) => FS.createDataFile(...args);

    var getUniqueRunDependency = id => id;

    var dependenciesPromise = null;

    var resolveRunDependencies = async () => dependenciesPromise;

    var runDependencies = 0;

    var dependenciesPromiseResolve = null;

    var removeRunDependency = id => {
      runDependencies--;
      Module['monitorRunDependencies']?.(runDependencies);
      if (!runDependencies) {
        dependenciesPromiseResolve();
      }
    };

    var addRunDependency = id => {
      if (!runDependencies) {
        dependenciesPromise =
            new Promise(resolve => dependenciesPromiseResolve = resolve);
      }
      runDependencies++;
      Module['monitorRunDependencies']?.(runDependencies);
    };

    var FS_handledByPreloadPlugin = async (byteArray, fullname) => {
      // Ensure plugins are ready.
      if (typeof Browser != 'undefined') Browser.init();
      for (var plugin of preloadPlugins) {
        if (plugin['canHandle'](fullname)) {
          return plugin['handle'](byteArray, fullname);
        }
      }
      // If no plugin handled this file then return the original/unmodified
      // byteArray.
      return byteArray;
    };

    var FS_preloadFile = async (
        parent, name, url, canRead, canWrite, dontCreateFile, canOwn,
        preFinish) => {
      // TODO we should allow people to just pass in a complete filename instead
      // of parent and name being that we just join them anyways
      var fullname = name ? PATH_FS.resolve(PATH.join2(parent, name)) : parent;
      var dep = getUniqueRunDependency(`cp ${fullname}`);
      // might have several active requests for the same fullname
      addRunDependency(dep);
      try {
        var byteArray = url;
        if (typeof url == 'string') {
          byteArray = await asyncLoad(url);
        }
        byteArray = await FS_handledByPreloadPlugin(byteArray, fullname);
        preFinish?.();
        if (!dontCreateFile) {
          FS_createDataFile(parent, name, byteArray, canRead, canWrite, canOwn);
        }
      } finally {
        removeRunDependency(dep);
      }
    };

    var FS_createPreloadedFile =
        (parent, name, url, canRead, canWrite, onload, onerror, dontCreateFile,
         canOwn, preFinish) => {
          FS_preloadFile(
              parent, name, url, canRead, canWrite, dontCreateFile, canOwn,
              preFinish)
              .then(onload)
              .catch(onerror);
        };

    var FS = {
      root: null,
      mounts: [],
      devices: {},
      streams: [],
      nextInode: 1,
      nameTable: null,
      currentPath: '/',
      initialized: false,
      ignorePermissions: true,
      filesystems: null,
      syncFSRequests: 0,
      ErrnoError: class {
        name = 'ErrnoError';
        // We set the `name` property to be able to identify `FS.ErrnoError`
        // - the `name` is a standard ECMA-262 property of error objects. Kind
        // of good to have it anyway.
        // - when using PROXYFS, an error can come from an underlying FS
        // as different FS objects have their own FS.ErrnoError each,
        // the test `err instanceof FS.ErrnoError` won't detect an error coming
        // from another filesystem, causing bugs. we'll use the reliable test
        // `err.name == "ErrnoError"` instead
        constructor(errno) {
          this.errno = errno;
        }
      },
      FSStream: class {
        shared = {};
        get object() {
          return this.node;
        }
        set object(val) {
          this.node = val;
        }
        get isRead() {
          return (this.flags & 2097155) !== 1;
        }
        get isWrite() {
          return (this.flags & 2097155) !== 0;
        }
        get isAppend() {
          return (this.flags & 1024);
        }
        get flags() {
          return this.shared.flags;
        }
        set flags(val) {
          this.shared.flags = val;
        }
        get position() {
          return this.shared.position;
        }
        set position(val) {
          this.shared.position = val;
        }
      },
      FSNode: class {
        node_ops = {};
        stream_ops = {};
        readMode = 292 | 73;
        writeMode = 146;
        mounted = null;
        constructor(parent, name, mode, rdev) {
          if (!parent) {
            parent = this;
          }
          this.parent = parent;
          this.mount = parent.mount;
          this.id = FS.nextInode++;
          this.name = name;
          this.mode = mode;
          this.rdev = rdev;
          this.atime = this.mtime = this.ctime = Date.now();
        }
        get read() {
          return (this.mode & this.readMode) === this.readMode;
        }
        set read(val) {
          val ? this.mode |= this.readMode : this.mode &= ~this.readMode;
        }
        get write() {
          return (this.mode & this.writeMode) === this.writeMode;
        }
        set write(val) {
          val ? this.mode |= this.writeMode : this.mode &= ~this.writeMode;
        }
        get isFolder() {
          return FS.isDir(this.mode);
        }
        get isDevice() {
          return FS.isChrdev(this.mode);
        }
        // The per-inode readiness wait-queue. The node carries a Set of
        // listener entries {cb}; producers (SOCKFS, PIPEFS) call
        // notifyListeners on a readiness transition, and poll()/epoll consume
        // it. It lives on the node (not the fd) so dup'd fds share one queue.
        // Only nodes that derive real readiness (sockets, pipes, and an epoll's
        // own node) ever use this - always-ready types (regular files, ttys)
        // never register or notify.
        addListener(cb, exclusive = false) {
          var entry = {cb, exclusive};
          var listeners = (this.listeners ??= new Set);
          listeners.add(entry);
          return {listeners, entry};
        }
        notifyListeners(flags) {
          // Iterates the set without copying, which is safe ONLY under a
          // load-bearing contract that every internal listener must honour:
          //   1. A listener must not run user code synchronously (a poll waiter
          //   only
          //      resolves a Promise; an epoll registration only re-lists +
          //      re-notifies; the epoll callback only schedules a tick). User
          //      code runs on a later tick, never inside this loop.
          //   2. A listener may delete entries only from ITS OWN waiter, never
          //   from
          //      a sibling node's set that may be mid-iteration. (Deleting an
          //      entry of the set being iterated here is fine - a Set tolerates
          //      removal of a not-yet-visited entry mid-iteration; mutating a
          //      *different* node's set is fine because that set is not being
          //      iterated.)
          // Violating either gives silently skipped wakeups that are
          // near-impossible to reproduce. Any new producer/listener must
          // preserve it.
          if (!this.listeners) return;
          // Fire every non-exclusive listener. Among EPOLLEXCLUSIVE
          // registrations (one fd watched by several epolls) wake only one,
          // rotating round-robin per node, to avoid a thundering herd. (Only
          // epoll registrations are ever exclusive; poll waiters and a node's
          // own consumers are not.)
          var excl;
          for (var entry of this.listeners) {
            if (entry.exclusive)
              (excl ||= []).push(entry);
            else
              entry.cb(flags);
          }
          if (excl) {
            var i = (this.exclTurn || 0) % excl.length;
            this.exclTurn = i + 1;
            excl[i].cb(flags);
          }
        }
      },
      lookupPath(path, opts = {}) {
        if (!path) {
          throw new FS.ErrnoError(44);
        }
        opts.follow_mount ??= true;
        if (!PATH.isAbs(path)) {
          path = FS.cwd() + '/' + path;
        }
        // limit max consecutive symlinks to SYMLOOP_MAX.
        linkloop: for (var nlinks = 0; nlinks < 40; nlinks++) {
          // split the absolute path
          var parts = path.split('/').filter(p => !!p);
          // start at the root
          var current = FS.root;
          var current_path = '/';
          for (var i = 0; i < parts.length; i++) {
            var islast = (i === parts.length - 1);
            if (islast && opts.parent) {
              // stop resolving
              break;
            }
            if (parts[i] === '.') {
              continue;
            }
            if (parts[i] === '..') {
              current_path = PATH.dirname(current_path);
              if (FS.isRoot(current)) {
                path = current_path + '/' + parts.slice(i + 1).join('/');
                // We're making progress here, don't let many consecutive ..'s
                // lead to ELOOP
                nlinks--;
                continue linkloop;
              } else {
                current = current.parent;
              }
              continue;
            }
            current_path = PATH.join2(current_path, parts[i]);
            try {
              current = FS.lookupNode(current, parts[i]);
            } catch (e) {
              // if noent_okay is true, suppress a ENOENT in the last component
              // and return an object with an undefined node. This is needed for
              // resolving symlinks in the path when creating a file.
              if ((e?.errno === 44) && islast && opts.noent_okay) {
                return {path: current_path};
              }
              throw e;
            }
            // jump to the mount's root node if this is a mountpoint
            if (FS.isMountpoint(current) && (!islast || opts.follow_mount)) {
              current = current.mounted.root;
            }
            // by default, lookupPath will not follow a symlink if it is the
            // final path component. setting opts.follow = true will override
            // this behavior.
            if (FS.isLink(current.mode) && (!islast || opts.follow)) {
              if (!current.node_ops.readlink) {
                throw new FS.ErrnoError(52);
              }
              var link = current.node_ops.readlink(current);
              if (!PATH.isAbs(link)) {
                link = PATH.dirname(current_path) + '/' + link;
              }
              path = link + '/' + parts.slice(i + 1).join('/');
              continue linkloop;
            }
          }
          return {path: current_path, node: current};
        }
        throw new FS.ErrnoError(32);
      },
      getPath(node) {
        var path;
        while (true) {
          if (FS.isRoot(node)) {
            var mount = node.mount.mountpoint;
            if (!path) return mount;
            return mount[mount.length - 1] !== '/' ? `${mount}/${path}` :
                                                     mount + path;
          }
          path = path ? `${node.name}/${path}` : node.name;
          node = node.parent;
        }
      },
      hashName(parentid, name) {
        var hash = 0;
        for (var i = 0; i < name.length; i++) {
          hash = ((hash << 5) - hash + name.charCodeAt(i)) | 0;
        }
        return ((parentid + hash) >>> 0) % FS.nameTable.length;
      },
      hashAddNode(node) {
        var hash = FS.hashName(node.parent.id, node.name);
        node.name_next = FS.nameTable[hash];
        FS.nameTable[hash] = node;
      },
      hashRemoveNode(node) {
        var hash = FS.hashName(node.parent.id, node.name);
        if (FS.nameTable[hash] === node) {
          FS.nameTable[hash] = node.name_next;
        } else {
          var current = FS.nameTable[hash];
          while (current) {
            if (current.name_next === node) {
              current.name_next = node.name_next;
              break;
            }
            current = current.name_next;
          }
        }
      },
      lookupNode(parent, name) {
        var errCode = FS.mayLookup(parent);
        if (errCode) {
          throw new FS.ErrnoError(errCode);
        }
        var hash = FS.hashName(parent.id, name);
        for (var node = FS.nameTable[hash]; node; node = node.name_next) {
          var nodeName = node.name;
          if (node.parent.id === parent.id && nodeName === name) {
            return node;
          }
        }
        // if we failed to find it in the cache, call into the VFS
        return FS.lookup(parent, name);
      },
      createNode(parent, name, mode, rdev) {
        var node = new FS.FSNode(parent, name, mode, rdev);
        FS.hashAddNode(node);
        return node;
      },
      destroyNode(node) {
        FS.hashRemoveNode(node);
      },
      isRoot(node) {
        return node === node.parent;
      },
      isMountpoint(node) {
        return !!node.mounted;
      },
      isFile(mode) {
        return (mode & 61440) === 32768;
      },
      isDir(mode) {
        return (mode & 61440) === 16384;
      },
      isLink(mode) {
        return (mode & 61440) === 40960;
      },
      isChrdev(mode) {
        return (mode & 61440) === 8192;
      },
      isBlkdev(mode) {
        return (mode & 61440) === 24576;
      },
      isFIFO(mode) {
        return (mode & 61440) === 4096;
      },
      isSocket(mode) {
        return (mode & 49152) === 49152;
      },
      flagsToPermissionString(flag) {
        var perms = ['r', 'w', 'rw'][flag & 3];
        if ((flag & 512)) {
          perms += 'w';
        }
        return perms;
      },
      nodePermissions(node, perms) {
        if (FS.ignorePermissions) {
          return 0;
        }
        // return 0 if any user, group or owner bits are set.
        if (perms.includes('r') && !(node.mode & 292)) {
          return 2;
        }
        if (perms.includes('w') && !(node.mode & 146)) {
          return 2;
        }
        if (perms.includes('x') && !(node.mode & 73)) {
          return 2;
        }
        return 0;
      },
      mayLookup(dir) {
        if (!FS.isDir(dir.mode)) return 54;
        var errCode = FS.nodePermissions(dir, 'x');
        if (errCode) return errCode;
        if (!dir.node_ops.lookup) return 2;
        return 0;
      },
      mayCreate(dir, name) {
        if (!FS.isDir(dir.mode)) {
          return 54;
        }
        try {
          var node = FS.lookupNode(dir, name);
          return 20;
        } catch (e) {
        }
        return FS.nodePermissions(dir, 'wx');
      },
      mayDelete(dir, name, isdir) {
        var node;
        try {
          node = FS.lookupNode(dir, name);
        } catch (e) {
          return e.errno;
        }
        var errCode = FS.nodePermissions(dir, 'wx');
        if (errCode) {
          return errCode;
        }
        if (isdir) {
          if (!FS.isDir(node.mode)) {
            return 54;
          }
          if (FS.isRoot(node) || FS.getPath(node) === FS.cwd()) {
            return 10;
          }
        } else if (FS.isDir(node.mode)) {
          return 31;
        }
        return 0;
      },
      mayOpen(node, flags) {
        if (!node) {
          return 44;
        }
        if (FS.isLink(node.mode)) {
          return 32;
        }
        var mode = FS.flagsToPermissionString(flags);
        if (FS.isDir(node.mode)) {
          // opening for write
          // TODO: check for O_SEARCH? (== search for dir only)
          if (mode !== 'r' || (flags & (512 | 64))) {
            return 31;
          }
        }
        return FS.nodePermissions(node, mode);
      },
      checkOpExists(op, err) {
        if (!op) {
          throw new FS.ErrnoError(err);
        }
        return op;
      },
      MAX_OPEN_FDS: 4096,
      nextfd() {
        for (var fd = 0; fd <= FS.MAX_OPEN_FDS; fd++) {
          if (!FS.streams[fd]) {
            return fd;
          }
        }
        throw new FS.ErrnoError(33);
      },
      getStreamChecked(fd) {
        var stream = FS.getStream(fd);
        if (!stream) {
          throw new FS.ErrnoError(8);
        }
        return stream;
      },
      getStream: fd => FS.streams[fd],
      createStream(stream, fd = -1) {
        // clone it, so we can return an instance of FSStream
        stream = Object.assign(new FS.FSStream, stream);
        if (fd == -1) {
          fd = FS.nextfd();
        }
        stream.fd = fd;
        FS.streams[fd] = stream;
        return stream;
      },
      closeStream(fd) {
        FS.streams[fd] = null;
      },
      dupStream(origStream, fd = -1) {
        var stream = FS.createStream(origStream, fd);
        stream.stream_ops?.dup?.(stream);
        return stream;
      },
      doSetAttr(stream, node, attr) {
        var setattr = stream?.stream_ops.setattr;
        var arg = setattr ? stream : node;
        setattr ??= node.node_ops.setattr;
        FS.checkOpExists(setattr, 63);
        try {
          setattr(arg, attr);
        } catch (e) {
          if (e instanceof RangeError) {
            throw new FS.ErrnoError(22);
          }
          throw e;
        }
      },
      chrdev_stream_ops: {
        open(stream) {
          var device = FS.getDevice(stream.node.rdev);
          // override node's stream ops with the device's
          stream.stream_ops = device.stream_ops;
          // forward the open call
          stream.stream_ops.open?.(stream);
        },
        llseek() {
          throw new FS.ErrnoError(70);
        }
      },
      major: dev => ((dev) >> 8),
      minor: dev => ((dev) & 255),
      makedev: (ma, mi) => ((ma) << 8 | (mi)),
      registerDevice(dev, ops) {
        FS.devices[dev] = {stream_ops: ops};
      },
      getDevice: dev => FS.devices[dev],
      getMounts(mount) {
        var mounts = [];
        var check = [mount];
        while (check.length) {
          var m = check.pop();
          mounts.push(m);
          check.push(...m.mounts);
        }
        return mounts;
      },
      syncfs(populate, callback) {
        if (typeof populate == 'function') {
          callback = populate;
          populate = false;
        }
        FS.syncFSRequests++;
        if (FS.syncFSRequests > 1) {
          err(`warning: ${
              FS.syncFSRequests} FS.syncfs operations in flight at once, probably just doing extra work`);
        }
        var mounts = FS.getMounts(FS.root.mount);
        var completed = 0;
        function doCallback(errCode) {
          FS.syncFSRequests--;
          return callback(errCode);
        }
        function done(errCode) {
          if (errCode) {
            if (!done.errored) {
              done.errored = true;
              return doCallback(errCode);
            }
            return;
          }
          if (++completed >= mounts.length) {
            doCallback(null);
          }
        }
        // sync all mounts
        for (var mount of mounts) {
          if (mount.type.syncfs) {
            mount.type.syncfs(mount, populate, done);
          } else {
            done(null);
          }
        }
      },
      mount(type, opts, mountpoint) {
        var root = mountpoint === '/';
        var pseudo = !mountpoint;
        var node;
        if (root && FS.root) {
          throw new FS.ErrnoError(10);
        } else if (!root && !pseudo) {
          var lookup = FS.lookupPath(mountpoint, {follow_mount: false});
          mountpoint = lookup.path;
          // use the absolute path
          node = lookup.node;
          if (FS.isMountpoint(node)) {
            throw new FS.ErrnoError(10);
          }
          if (!FS.isDir(node.mode)) {
            throw new FS.ErrnoError(54);
          }
        }
        var mount = {type, opts, mountpoint, mounts: []};
        // create a root node for the fs
        var mountRoot = type.mount(mount);
        mountRoot.mount = mount;
        mount.root = mountRoot;
        if (root) {
          FS.root = mountRoot;
        } else if (node) {
          // set as a mountpoint
          node.mounted = mount;
          // add the new mount to the current mount's children
          if (node.mount) {
            node.mount.mounts.push(mount);
          }
        }
        return mountRoot;
      },
      unmount(mountpoint) {
        var lookup = FS.lookupPath(mountpoint, {follow_mount: false});
        if (!FS.isMountpoint(lookup.node)) {
          throw new FS.ErrnoError(28);
        }
        // destroy the nodes for this mount, and all its child mounts
        var node = lookup.node;
        var mount = node.mounted;
        var mounts = FS.getMounts(mount);
        for (var [hash, current] of Object.entries(FS.nameTable)) {
          while (current) {
            var next = current.name_next;
            if (mounts.includes(current.mount)) {
              FS.destroyNode(current);
            }
            current = next;
          }
        }
        // no longer a mountpoint
        node.mounted = null;
        // remove this mount from the child mounts
        var idx = node.mount.mounts.indexOf(mount);
        node.mount.mounts.splice(idx, 1);
      },
      lookup(parent, name) {
        return parent.node_ops.lookup(parent, name);
      },
      mknod(path, mode, dev) {
        var lookup = FS.lookupPath(path, {parent: true});
        var parent = lookup.node;
        var name = PATH.basename(path);
        if (!name) {
          throw new FS.ErrnoError(28);
        }
        if (name === '.' || name === '..') {
          throw new FS.ErrnoError(20);
        }
        var errCode = FS.mayCreate(parent, name);
        if (errCode) {
          throw new FS.ErrnoError(errCode);
        }
        if (!parent.node_ops.mknod) {
          throw new FS.ErrnoError(63);
        }
        return parent.node_ops.mknod(parent, name, mode, dev);
      },
      statfs(path) {
        return FS.statfsNode(FS.lookupPath(path, {follow: true}).node);
      },
      statfsStream(stream) {
        // We keep a separate statfsStream function because noderawfs overrides
        // it. In noderawfs, stream.node is sometimes null. Instead, we need to
        // look at stream.path.
        return FS.statfsNode(stream.node);
      },
      statfsNode(node) {
        // NOTE: None of the defaults here are true. We're just returning safe
        // and
        //       sane values. Currently nodefs and rawfs replace these defaults,
        //       other file systems leave them alone.
        var rtn = {
          bsize: 4096,
          frsize: 4096,
          blocks: 1e6,
          bfree: 5e5,
          bavail: 5e5,
          files: FS.nextInode,
          ffree: FS.nextInode - 1,
          fsid: 42,
          flags: 2,
          namelen: 255
        };
        if (node.node_ops.statfs) {
          Object.assign(rtn, node.node_ops.statfs(node.mount.opts.root));
        }
        return rtn;
      },
      create(path, mode = 438) {
        mode &= 4095;
        mode |= 32768;
        return FS.mknod(path, mode, 0);
      },
      mkdir(path, mode = 511) {
        mode &= 511 | 512;
        mode |= 16384;
        return FS.mknod(path, mode, 0);
      },
      mkdirTree(path, mode) {
        var dirs = path.split('/');
        var d = '';
        for (var dir of dirs) {
          if (!dir) continue;
          if (d || PATH.isAbs(path)) d += '/';
          d += dir;
          try {
            FS.mkdir(d, mode);
          } catch (e) {
            if (e.errno != 20) throw e;
          }
        }
      },
      mkdev(path, mode, dev) {
        if (typeof dev == 'undefined') {
          dev = mode;
          mode = 438;
        }
        mode |= 8192;
        return FS.mknod(path, mode, dev);
      },
      symlink(oldpath, newpath) {
        if (!PATH_FS.resolve(oldpath)) {
          throw new FS.ErrnoError(44);
        }
        var lookup = FS.lookupPath(newpath, {parent: true});
        var parent = lookup.node;
        if (!parent) {
          throw new FS.ErrnoError(44);
        }
        var newname = PATH.basename(newpath);
        var errCode = FS.mayCreate(parent, newname);
        if (errCode) {
          throw new FS.ErrnoError(errCode);
        }
        if (!parent.node_ops.symlink) {
          throw new FS.ErrnoError(63);
        }
        return parent.node_ops.symlink(parent, newname, oldpath);
      },
      link(oldpath, newpath, flags) {
        var lookup = FS.lookupPath(newpath, {parent: true});
        var parent = lookup.node;
        if (!parent) {
          throw new FS.ErrnoError(44);
        }
        var newname = PATH.basename(newpath);
        var errCode = FS.mayCreate(parent, newname);
        if (errCode) {
          throw new FS.ErrnoError(errCode);
        }
        // Hardlinks are only supported by filesystem backends that provide a
        // `link` node op (e.g. NODERAWFS backed by the host). NODEFS omits it:
        // a host hardlink cannot be confined to the mount root.
        if (!parent.node_ops.link) {
          throw new FS.ErrnoError(34);
        }
        return parent.node_ops.link(parent, newname, oldpath, flags);
      },
      rename(old_path, new_path) {
        var old_dirname = PATH.dirname(old_path);
        var new_dirname = PATH.dirname(new_path);
        var old_name = PATH.basename(old_path);
        var new_name = PATH.basename(new_path);
        // parents must exist
        var lookup, old_dir, new_dir;
        // let the errors from non existent directories percolate up
        lookup = FS.lookupPath(old_path, {parent: true});
        old_dir = lookup.node;
        lookup = FS.lookupPath(new_path, {parent: true});
        new_dir = lookup.node;
        if (!old_dir || !new_dir) throw new FS.ErrnoError(44);
        // need to be part of the same mount
        if (old_dir.mount !== new_dir.mount) {
          throw new FS.ErrnoError(75);
        }
        // source must exist
        var old_node = FS.lookupNode(old_dir, old_name);
        // old path should not be an ancestor of the new path
        var relative = PATH_FS.relative(old_path, new_dirname);
        if (relative.charAt(0) !== '.') {
          throw new FS.ErrnoError(28);
        }
        // new path should not be an ancestor of the old path
        relative = PATH_FS.relative(new_path, old_dirname);
        if (relative.charAt(0) !== '.') {
          throw new FS.ErrnoError(55);
        }
        // see if the new path already exists
        var new_node;
        try {
          new_node = FS.lookupNode(new_dir, new_name);
        } catch (e) {
        }
        // early out if nothing needs to change
        if (old_node === new_node) {
          return;
        }
        // we'll need to delete the old entry
        var isdir = FS.isDir(old_node.mode);
        var errCode = FS.mayDelete(old_dir, old_name, isdir);
        if (errCode) {
          throw new FS.ErrnoError(errCode);
        }
        // need delete permissions if we'll be overwriting.
        // need create permissions if new doesn't already exist.
        errCode = new_node ? FS.mayDelete(new_dir, new_name, isdir) :
                             FS.mayCreate(new_dir, new_name);
        if (errCode) {
          throw new FS.ErrnoError(errCode);
        }
        if (!old_dir.node_ops.rename) {
          throw new FS.ErrnoError(63);
        }
        if (FS.isMountpoint(old_node) ||
            (new_node && FS.isMountpoint(new_node))) {
          throw new FS.ErrnoError(10);
        }
        // if we are going to change the parent, check write permissions
        if (new_dir !== old_dir) {
          errCode = FS.nodePermissions(old_dir, 'w');
          if (errCode) {
            throw new FS.ErrnoError(errCode);
          }
        }
        // remove the node from the lookup hash
        FS.hashRemoveNode(old_node);
        // do the underlying fs rename
        try {
          old_dir.node_ops.rename(old_node, new_dir, new_name);
          // update old node (we do this here to avoid each backend
          // needing to)
          old_node.parent = new_dir;
        } catch (e) {
          throw e;
        } finally {
          // add the node back to the hash (in case node_ops.rename
          // changed its name)
          FS.hashAddNode(old_node);
        }
      },
      rmdir(path) {
        var lookup = FS.lookupPath(path, {parent: true});
        var parent = lookup.node;
        var name = PATH.basename(path);
        var node = FS.lookupNode(parent, name);
        var errCode = FS.mayDelete(parent, name, true);
        if (errCode) {
          throw new FS.ErrnoError(errCode);
        }
        if (!parent.node_ops.rmdir) {
          throw new FS.ErrnoError(63);
        }
        if (FS.isMountpoint(node)) {
          throw new FS.ErrnoError(10);
        }
        parent.node_ops.rmdir(parent, name);
        FS.destroyNode(node);
      },
      readdir(path) {
        var lookup = FS.lookupPath(path, {follow: true});
        var node = lookup.node;
        var readdir = FS.checkOpExists(node.node_ops.readdir, 54);
        return readdir(node);
      },
      unlink(path) {
        var lookup = FS.lookupPath(path, {parent: true});
        var parent = lookup.node;
        if (!parent) {
          throw new FS.ErrnoError(44);
        }
        var name = PATH.basename(path);
        var node = FS.lookupNode(parent, name);
        var errCode = FS.mayDelete(parent, name, false);
        if (errCode) {
          // According to POSIX, we should map EISDIR to EPERM, but
          // we instead do what Linux does (and we must, as we use
          // the musl linux libc).
          throw new FS.ErrnoError(errCode);
        }
        if (!parent.node_ops.unlink) {
          throw new FS.ErrnoError(63);
        }
        if (FS.isMountpoint(node)) {
          throw new FS.ErrnoError(10);
        }
        parent.node_ops.unlink(parent, name);
        FS.destroyNode(node);
      },
      readlink(path) {
        var lookup = FS.lookupPath(path);
        var link = lookup.node;
        if (!link) {
          throw new FS.ErrnoError(44);
        }
        if (!link.node_ops.readlink) {
          throw new FS.ErrnoError(28);
        }
        return link.node_ops.readlink(link);
      },
      stat(path, dontFollow) {
        var lookup = FS.lookupPath(path, {follow: !dontFollow});
        var node = lookup.node;
        var getattr = FS.checkOpExists(node.node_ops.getattr, 63);
        return getattr(node);
      },
      fstat(fd) {
        var stream = FS.getStreamChecked(fd);
        var node = stream.node;
        var getattr = stream.stream_ops.getattr;
        var arg = getattr ? stream : node;
        getattr ??= node.node_ops.getattr;
        FS.checkOpExists(getattr, 63);
        return getattr(arg);
      },
      lstat(path) {
        return FS.stat(path, true);
      },
      doChmod(stream, node, mode, dontFollow) {
        FS.doSetAttr(stream, node, {
          mode: (mode & 4095) | (node.mode & ~4095),
          ctime: Date.now(),
          dontFollow
        });
      },
      chmod(path, mode, dontFollow) {
        var node;
        if (typeof path == 'string') {
          var lookup = FS.lookupPath(path, {follow: !dontFollow});
          node = lookup.node;
        } else {
          node = path;
        }
        FS.doChmod(null, node, mode, dontFollow);
      },
      lchmod(path, mode) {
        FS.chmod(path, mode, true);
      },
      fchmod(fd, mode) {
        var stream = FS.getStreamChecked(fd);
        FS.doChmod(stream, stream.node, mode, false);
      },
      doChown(stream, node, dontFollow) {
        FS.doSetAttr(stream, node, {timestamp: Date.now(), dontFollow});
      },
      chown(path, uid, gid, dontFollow) {
        var node;
        if (typeof path == 'string') {
          var lookup = FS.lookupPath(path, {follow: !dontFollow});
          node = lookup.node;
        } else {
          node = path;
        }
        FS.doChown(null, node, dontFollow);
      },
      lchown(path, uid, gid) {
        FS.chown(path, uid, gid, true);
      },
      fchown(fd, uid, gid) {
        var stream = FS.getStreamChecked(fd);
        FS.doChown(stream, stream.node, false);
      },
      doTruncate(stream, node, len) {
        if (FS.isDir(node.mode)) {
          throw new FS.ErrnoError(31);
        }
        if (!FS.isFile(node.mode)) {
          throw new FS.ErrnoError(28);
        }
        var errCode = FS.nodePermissions(node, 'w');
        if (errCode) {
          throw new FS.ErrnoError(errCode);
        }
        FS.doSetAttr(stream, node, {size: len, timestamp: Date.now()});
      },
      truncate(path, len) {
        if (len < 0) {
          throw new FS.ErrnoError(28);
        }
        var node;
        if (typeof path == 'string') {
          var lookup = FS.lookupPath(path, {follow: true});
          node = lookup.node;
        } else {
          node = path;
        }
        FS.doTruncate(null, node, len);
      },
      ftruncate(fd, len) {
        var stream = FS.getStreamChecked(fd);
        if (len < 0 || (stream.flags & 2097155) === 0) {
          throw new FS.ErrnoError(28);
        }
        FS.doTruncate(stream, stream.node, len);
      },
      utime(path, atime, mtime, dontFollow) {
        var lookup = FS.lookupPath(path, {follow: !dontFollow});
        FS.doSetAttr(null, lookup.node, {atime, mtime, dontFollow});
      },
      open(path, flags, mode = 438) {
        if (path === '') {
          throw new FS.ErrnoError(44);
        }
        flags = FS_modeStringToFlags(flags);
        if ((flags & 64)) {
          mode = (mode & 4095) | 32768;
        } else {
          mode = 0;
        }
        var node;
        var isDirPath;
        if (typeof path == 'object') {
          node = path;
        } else {
          isDirPath = path.endsWith('/');
          // noent_okay makes it so that if the final component of the path
          // doesn't exist, lookupPath returns `node: undefined`. `path` will be
          // updated to point to the target of all symlinks.
          var lookup = FS.lookupPath(
              path, {follow: !(flags & 131072), noent_okay: true});
          node = lookup.node;
          path = lookup.path;
        }
        // perhaps we need to create the node
        var created = false;
        if ((flags & 64)) {
          if (node) {
            // if O_CREAT and O_EXCL are set, error out if the node already
            // exists
            if ((flags & 128)) {
              throw new FS.ErrnoError(20);
            }
          } else if (isDirPath) {
            throw new FS.ErrnoError(31);
          } else {
            // node doesn't exist, try to create it
            // Ignore the permission bits here to ensure we can `open` this new
            // file below. We use chmod below to apply the permissions once the
            // file is open.
            node = FS.mknod(path, mode | 511, 0);
            created = true;
          }
        }
        if (!node) {
          throw new FS.ErrnoError(44);
        }
        // can't truncate a device
        if (FS.isChrdev(node.mode)) {
          flags &= ~512;
        }
        // if asked only for a directory, then this must be one
        if ((flags & 65536) && !FS.isDir(node.mode)) {
          throw new FS.ErrnoError(54);
        }
        // check permissions, if this is not a file we just created now (it is
        // ok to create and write to a file with read-only permissions; it is
        // read-only for later use)
        if (!created) {
          var errCode = FS.mayOpen(node, flags);
          if (errCode) {
            throw new FS.ErrnoError(errCode);
          }
        }
        // do truncation if necessary
        if ((flags & 512) && !created) {
          FS.truncate(node, 0);
        }
        // we've already handled these, don't pass down to the underlying vfs
        flags &= ~(128 | 512 | 131072);
        // register the stream with the filesystem
        var stream = FS.createStream({
          node,
          path: FS.getPath(node),
          // we want the absolute path to the node
          flags,
          seekable: true,
          position: 0,
          stream_ops: node.stream_ops,
          // used by the file family libc calls (fopen, fwrite, ferror, etc.)
          ungotten: [],
          error: false
        });
        // call the new stream's open function
        if (stream.stream_ops.open) {
          stream.stream_ops.open(stream);
        }
        if (created) {
          FS.chmod(node, mode & 511);
        }
        return stream;
      },
      close(stream) {
        if (FS.isClosed(stream)) {
          throw new FS.ErrnoError(8);
        }
        if (stream.getdents) stream.getdents = null;
        // free readdir state
        // The fd is going away: wake anything waiting on it (poll/epoll) with
        // POLLNVAL so a blocking wait unblocks and an epoll registration is
        // evicted on its next derive. Only sockets/pipes/epoll ever carry a
        // wait-queue, so for every other stream (incl. nodeless noderawfs
        // stdio) this is a no-op.
        stream.node?.notifyListeners(32);
        try {
          if (stream.stream_ops.close) {
            stream.stream_ops.close(stream);
          }
        } catch (e) {
          throw e;
        } finally {
          FS.closeStream(stream.fd);
        }
        stream.fd = null;
      },
      isClosed(stream) {
        return stream.fd === null;
      },
      llseek(stream, offset, whence) {
        if (FS.isClosed(stream)) {
          throw new FS.ErrnoError(8);
        }
        if (!stream.seekable || !stream.stream_ops.llseek) {
          throw new FS.ErrnoError(70);
        }
        if (whence != 0 && whence != 1 && whence != 2) {
          throw new FS.ErrnoError(28);
        }
        stream.position = stream.stream_ops.llseek(stream, offset, whence);
        stream.ungotten = [];
        return stream.position;
      },
      read(stream, buffer, offset, length, position) {
        if (length < 0 || position < 0) {
          throw new FS.ErrnoError(28);
        }
        if (FS.isClosed(stream)) {
          throw new FS.ErrnoError(8);
        }
        if ((stream.flags & 2097155) === 1) {
          throw new FS.ErrnoError(8);
        }
        if (FS.isDir(stream.node.mode)) {
          throw new FS.ErrnoError(31);
        }
        if (!stream.stream_ops.read) {
          throw new FS.ErrnoError(28);
        }
        var seeking = typeof position != 'undefined';
        if (!seeking) {
          position = stream.position;
        } else if (!stream.seekable) {
          throw new FS.ErrnoError(70);
        }
        var bytesRead =
            stream.stream_ops.read(stream, buffer, offset, length, position);
        if (!seeking) stream.position += bytesRead;
        return bytesRead;
      },
      write(stream, buffer, offset, length, position, canOwn) {
        if (length < 0 || position < 0) {
          throw new FS.ErrnoError(28);
        }
        if (FS.isClosed(stream)) {
          throw new FS.ErrnoError(8);
        }
        if ((stream.flags & 2097155) === 0) {
          throw new FS.ErrnoError(8);
        }
        if (FS.isDir(stream.node.mode)) {
          throw new FS.ErrnoError(31);
        }
        if (!stream.stream_ops.write) {
          throw new FS.ErrnoError(28);
        }
        if (stream.seekable && stream.flags & 1024) {
          // seek to the end before writing in append mode
          FS.llseek(stream, 0, 2);
        }
        var seeking = typeof position != 'undefined';
        if (!seeking) {
          position = stream.position;
        } else if (!stream.seekable) {
          throw new FS.ErrnoError(70);
        }
        var bytesWritten = stream.stream_ops.write(
            stream, buffer, offset, length, position, canOwn);
        if (!seeking) stream.position += bytesWritten;
        return bytesWritten;
      },
      mmap(stream, length, position, prot, flags) {
        // User requests writing to file (prot & PROT_WRITE != 0).
        // Checking if we have permissions to write to the file unless
        // MAP_PRIVATE flag is set. According to POSIX spec it is possible
        // to write to file opened in read-only mode with MAP_PRIVATE flag,
        // as all modifications will be visible only in the memory of
        // the current process.
        if ((prot & 2) !== 0 && (flags & 2) === 0 &&
            (stream.flags & 2097155) !== 2) {
          throw new FS.ErrnoError(2);
        }
        if ((stream.flags & 2097155) === 1) {
          throw new FS.ErrnoError(2);
        }
        if (!stream.stream_ops.mmap) {
          throw new FS.ErrnoError(43);
        }
        if (!length) {
          throw new FS.ErrnoError(28);
        }
        return stream.stream_ops.mmap(stream, length, position, prot, flags);
      },
      msync(stream, buffer, offset, length, mmapFlags) {
        if (!stream.stream_ops.msync) {
          return 0;
        }
        return stream.stream_ops.msync(
            stream, buffer, offset, length, mmapFlags);
      },
      ioctl(stream, cmd, arg) {
        if (!stream.stream_ops.ioctl) {
          throw new FS.ErrnoError(59);
        }
        return stream.stream_ops.ioctl(stream, cmd, arg);
      },
      readFile(path, opts = {}) {
        opts.flags = opts.flags ?? 0;
        opts.encoding = opts.encoding ?? 'binary';
        if (opts.encoding !== 'utf8' && opts.encoding !== 'binary') {
          abort(`Invalid encoding type "${opts.encoding}"`);
        }
        var stream = FS.open(path, opts.flags);
        var stat = FS.stat(path);
        var length = stat.size;
        var buf = new Uint8Array(length);
        FS.read(stream, buf, 0, length, 0);
        if (opts.encoding === 'utf8') {
          buf = UTF8ArrayToString(buf);
        }
        FS.close(stream);
        return buf;
      },
      writeFile(path, data, opts = {}) {
        opts.flags = opts.flags ?? 577;
        var stream = FS.open(path, opts.flags, opts.mode);
        data = FS_fileDataToTypedArray(data);
        FS.write(stream, data, 0, data.byteLength, undefined, opts.canOwn);
        FS.close(stream);
      },
      cwd: () => FS.currentPath,
      chdir(path) {
        var lookup = FS.lookupPath(path, {follow: true});
        if (lookup.node === null) {
          throw new FS.ErrnoError(44);
        }
        if (!FS.isDir(lookup.node.mode)) {
          throw new FS.ErrnoError(54);
        }
        var errCode = FS.nodePermissions(lookup.node, 'x');
        if (errCode) {
          throw new FS.ErrnoError(errCode);
        }
        FS.currentPath = lookup.path;
      },
      createDefaultDirectories() {
        FS.mkdir('/tmp');
        FS.mkdir('/home');
        FS.mkdir('/home/web_user');
      },
      createDefaultDevices() {
        // create /dev
        FS.mkdir('/dev');
        // setup /dev/null
        FS.registerDevice(FS.makedev(1, 3), {
          read: () => 0,
          write: (stream, buffer, offset, length, pos) => length,
          llseek: () => 0
        });
        FS.mkdev('/dev/null', FS.makedev(1, 3));
        // setup /dev/tty and /dev/tty1
        // stderr needs to print output using err() rather than out()
        // so we register a second tty just for it.
        TTY.register(FS.makedev(5, 0), TTY.default_tty_ops);
        TTY.register(FS.makedev(6, 0), TTY.default_tty1_ops);
        FS.mkdev('/dev/tty', FS.makedev(5, 0));
        FS.mkdev('/dev/tty1', FS.makedev(6, 0));
        // setup /dev/[u]random
        // use a buffer to avoid overhead of individual crypto calls per byte
        var randomBuffer = new Uint8Array(1024), randomLeft = 0;
        var randomByte = () => {
          if (randomLeft === 0) {
            randomFill(randomBuffer);
            randomLeft = randomBuffer.byteLength;
          }
          return randomBuffer[--randomLeft];
        };
        FS.createDevice('/dev', 'random', randomByte);
        FS.createDevice('/dev', 'urandom', randomByte);
        // we're not going to emulate the actual shm device,
        // just create the tmp dirs that reside in it commonly
        FS.mkdir('/dev/shm');
        FS.mkdir('/dev/shm/tmp');
      },
      createSpecialDirectories() {
        // create /proc/self/fd which allows /proc/self/fd/6 => readlink gives
        // the name of the stream for fd 6 (see test_unistd_ttyname)
        FS.mkdir('/proc');
        var proc_self = FS.mkdir('/proc/self');
        FS.mkdir('/proc/self/fd');
        FS.mount(
            {
              mount() {
                var node = FS.createNode(proc_self, 'fd', 16895, 73);
                node.stream_ops = {llseek: MEMFS.stream_ops.llseek};
                node.node_ops = {
                  lookup(parent, name) {
                    var fd = +name;
                    var stream = FS.getStreamChecked(fd);
                    var ret = {
                      parent: null,
                      mount: {mountpoint: 'fake'},
                      node_ops: {readlink: () => stream.path},
                      id: fd + 1
                    };
                    ret.parent = ret;
                    // make it look like a simple root node
                    return ret;
                  },
                  readdir() {
                    return Array.from(FS.streams.entries())
                        .filter(([k, v]) => v)
                        .map(([k, v]) => k.toString());
                  }
                };
                return node;
              }
            },
            {}, '/proc/self/fd');
      },
      createStandardStreams(input, output, error) {
        // TODO deprecate the old functionality of a single
        // input / output callback and that utilizes FS.createDevice
        // and instead require a unique set of stream ops
        // by default, we symlink the standard streams to the
        // default tty devices. however, if the standard streams
        // have been overwritten we create a unique device for
        // them instead.
        if (input) {
          FS.createDevice('/dev', 'stdin', input);
        } else {
          FS.symlink('/dev/tty', '/dev/stdin');
        }
        if (output) {
          FS.createDevice('/dev', 'stdout', null, output);
        } else {
          FS.symlink('/dev/tty', '/dev/stdout');
        }
        if (error) {
          FS.createDevice('/dev', 'stderr', null, error);
        } else {
          FS.symlink('/dev/tty1', '/dev/stderr');
        }
        // open default streams for the stdin, stdout and stderr devices
        var stdin = FS.open('/dev/stdin', 0);
        var stdout = FS.open('/dev/stdout', 1);
        var stderr = FS.open('/dev/stderr', 1);
      },
      staticInit() {
        FS.nameTable = new Array(4096);
        FS.mount(MEMFS, {}, '/');
        FS.createDefaultDirectories();
        FS.createDefaultDevices();
        FS.createSpecialDirectories();
        FS.filesystems = {'MEMFS': MEMFS};
      },
      init(input, output, error) {
        FS.initialized = true;
        // Allow Module.stdin etc. to provide defaults, if none explicitly
        // passed to us here
        input ??= Module['stdin'];
        output ??= Module['stdout'];
        error ??= Module['stderr'];
        FS.createStandardStreams(input, output, error);
      },
      quit() {
        FS.initialized = false;
        // force-flush all streams, so we get musl std streams printed out
        // close all of our streams
        for (var stream of FS.streams) {
          if (stream) {
            FS.close(stream);
          }
        }
      },
      findObject(path, dontResolveLastLink) {
        var ret = FS.analyzePath(path, dontResolveLastLink);
        if (!ret.exists) {
          return null;
        }
        return ret.object;
      },
      analyzePath(path, dontResolveLastLink) {
        // operate from within the context of the symlink's target
        try {
          var lookup = FS.lookupPath(path, {follow: !dontResolveLastLink});
          path = lookup.path;
        } catch (e) {
        }
        var ret = {
          isRoot: false,
          exists: false,
          error: 0,
          name: null,
          path: null,
          object: null,
          parentExists: false,
          parentPath: null,
          parentObject: null
        };
        try {
          var lookup = FS.lookupPath(path, {parent: true});
          ret.parentExists = true;
          ret.parentPath = lookup.path;
          ret.parentObject = lookup.node;
          ret.name = PATH.basename(path);
          lookup = FS.lookupPath(path, {follow: !dontResolveLastLink});
          ret.exists = true;
          ret.path = lookup.path;
          ret.object = lookup.node;
          ret.name = lookup.node.name;
          ret.isRoot = lookup.path === '/';
        } catch (e) {
          ret.error = e.errno;
        }
        return ret;
      },
      createPath(parent, path, canRead, canWrite) {
        parent = typeof parent == 'string' ? parent : FS.getPath(parent);
        var parts = path.split('/').reverse();
        while (parts.length) {
          var part = parts.pop();
          if (!part) continue;
          var current = PATH.join2(parent, part);
          try {
            FS.mkdir(current);
          } catch (e) {
            if (e.errno != 20) throw e;
          }
          parent = current;
        }
        return current;
      },
      createFile(parent, name, properties, canRead, canWrite) {
        var path = PATH.join2(
            typeof parent == 'string' ? parent : FS.getPath(parent), name);
        var mode = FS_getMode(canRead, canWrite);
        return FS.create(path, mode);
      },
      createDataFile(parent, name, data, canRead, canWrite, canOwn) {
        var path = name;
        if (parent) {
          parent = typeof parent == 'string' ? parent : FS.getPath(parent);
          path = name ? PATH.join2(parent, name) : parent;
        }
        var mode = FS_getMode(canRead, canWrite);
        var node = FS.create(path, mode);
        if (data) {
          data = FS_fileDataToTypedArray(data);
          // make sure we can write to the file
          FS.chmod(node, mode | 146);
          var stream = FS.open(node, 577);
          FS.write(stream, data, 0, data.length, 0, canOwn);
          FS.close(stream);
          FS.chmod(node, mode);
        }
      },
      createDevice(parent, name, input, output) {
        var path = PATH.join2(
            typeof parent == 'string' ? parent : FS.getPath(parent), name);
        var mode = FS_getMode(!!input, !!output);
        FS.createDevice.major ??= 64;
        var dev = FS.makedev(FS.createDevice.major++, 0);
        // Create a fake device that a set of stream ops to emulate
        // the old behavior.
        FS.registerDevice(dev, {
          open(stream) {
            stream.seekable = false;
          },
          close(stream) {
            // flush any pending line data
            if (output?.buffer?.length) {
              output(10);
            }
          },
          read(stream, buffer, offset, length, pos) {
            var bytesRead = 0;
            for (var i = 0; i < length; i++) {
              var result;
              try {
                result = input();
              } catch (e) {
                throw new FS.ErrnoError(29);
              }
              if (result === undefined && bytesRead === 0) {
                throw new FS.ErrnoError(6);
              }
              if (result === null || result === undefined) break;
              bytesRead++;
              buffer[offset + i] = result;
            }
            if (bytesRead) {
              stream.node.atime = Date.now();
            }
            return bytesRead;
          },
          write(stream, buffer, offset, length, pos) {
            for (var i = 0; i < length; i++) {
              try {
                output(buffer[offset + i]);
              } catch (e) {
                throw new FS.ErrnoError(29);
              }
            }
            if (length) {
              stream.node.mtime = stream.node.ctime = Date.now();
            }
            return i;
          }
        });
        return FS.mkdev(path, mode, dev);
      },
      forceLoadFile(obj) {
        if (obj.isDevice || obj.isFolder || obj.link || obj.contents)
          return true;
        if (globalThis.XMLHttpRequest) {
          abort(
              'Lazy loading should have been performed (contents set) in createLazyFile, but it was not. Lazy loading only works in web workers. Use --embed-file or --preload-file in emcc on the main thread.');
        } else {
          // Command-line.
          try {
            obj.contents = readBinary(obj.url);
          } catch (e) {
            throw new FS.ErrnoError(29);
          }
        }
      },
      createLazyFile(parent, name, url, canRead, canWrite) {
        // Lazy chunked Uint8Array (implements get and length from Uint8Array).
        // Actual getting is abstracted away for eventual reuse.
        class LazyUint8Array {
          lengthKnown = false;
          chunks = [];
          // Loaded chunks. Index is the chunk number
          get(idx) {
            if (idx > this.length - 1 || idx < 0) {
              return undefined;
            }
            var chunkOffset = idx % this.chunkSize;
            var chunkNum = (idx / this.chunkSize) | 0;
            return this.getter(chunkNum)[chunkOffset];
          }
          setDataGetter(getter) {
            this.getter = getter;
          }
          cacheLength() {
            // Find length
            var xhr = new XMLHttpRequest;
            xhr.open('HEAD', url, false);
            xhr.send(null);
            if (!(xhr.status >= 200 && xhr.status < 300 || xhr.status === 304))
              abort(`Couldn't load ${url}. Status: ${xhr.status}`);
            var datalength = Number(xhr.getResponseHeader('Content-length'));
            var header;
            var hasByteServing =
                (header = xhr.getResponseHeader('Accept-Ranges')) &&
                header === 'bytes';
            var usesGzip =
                (header = xhr.getResponseHeader('Content-Encoding')) &&
                header === 'gzip';
            var chunkSize = 1024 * 1024;
            // Chunk size in bytes
            if (!hasByteServing) chunkSize = datalength;
            // Function to get a range from the remote URL.
            var doXHR = (from, to) => {
              if (from > to)
                abort(`invalid range (${from}, ${to}) or no bytes requested!`);
              if (to > datalength - 1)
                abort(`only ${datalength} bytes available! programmer error!`);
              // TODO: Use mozResponseArrayBuffer, responseStream, etc. if
              // available.
              var xhr = new XMLHttpRequest;
              xhr.open('GET', url, false);
              if (datalength !== chunkSize)
                xhr.setRequestHeader('Range', `bytes=${from}-${to}`);
              // Some hints to the browser that we want binary data.
              xhr.responseType = 'arraybuffer';
              if (xhr.overrideMimeType) {
                xhr.overrideMimeType('text/plain; charset=x-user-defined');
              }
              xhr.send(null);
              if (!(xhr.status >= 200 && xhr.status < 300 ||
                    xhr.status === 304))
                abort(`Couldn't load ${url}. Status: ${xhr.status}`);
              if (xhr.response !== undefined) {
                return new Uint8Array(
                    /** @type{Array<number>} */ (xhr.response || []));
              }
              return intArrayFromString(xhr.responseText ?? '', true);
            };
            var lazyArray = this;
            lazyArray.setDataGetter(chunkNum => {
              var start = chunkNum * chunkSize;
              var end = (chunkNum + 1) * chunkSize - 1;
              // including this byte
              end = Math.min(end, datalength - 1);
              // if datalength-1 is selected, this is the last block
              if (typeof lazyArray.chunks[chunkNum] == 'undefined') {
                lazyArray.chunks[chunkNum] = doXHR(start, end);
              }
              if (typeof lazyArray.chunks[chunkNum] == 'undefined')
                abort('doXHR failed!');
              return lazyArray.chunks[chunkNum];
            });
            if (usesGzip || !datalength) {
              // if the server uses gzip or doesn't supply the length, we have
              // to download the whole file to get the (uncompressed) length
              chunkSize = datalength = 1;
              // this will force getter(0)/doXHR do download the whole file
              datalength = this.getter(0).length;
              chunkSize = datalength;
              out('LazyFiles on gzip forces download of the whole file when length is accessed');
            }
            this._length = datalength;
            this._chunkSize = chunkSize;
            this.lengthKnown = true;
          }
          get length() {
            if (!this.lengthKnown) {
              this.cacheLength();
            }
            return this._length;
          }
          get chunkSize() {
            if (!this.lengthKnown) {
              this.cacheLength();
            }
            return this._chunkSize;
          }
        }
        if (globalThis.XMLHttpRequest) {
          if (!ENVIRONMENT_IS_WORKER)
            abort(
                'Cannot do synchronous binary XHRs outside webworkers in modern browsers. Use --embed-file or --preload-file in emcc');
          var lazyArray = new LazyUint8Array;
          var properties = {isDevice: false, contents: lazyArray};
        } else {
          var properties = {isDevice: false, url};
        }
        var node = FS.createFile(parent, name, properties, canRead, canWrite);
        // This is a total hack, but I want to get this lazy file code out of
        // the core of MEMFS. If we want to keep this lazy file concept I feel
        // it should be its own thin LAZYFS proxying calls to MEMFS.
        if (properties.contents) {
          node.contents = properties.contents;
        } else if (properties.url) {
          node.contents = null;
          node.url = properties.url;
        }
        // Add a function that defers querying the file size until it is asked
        // the first time.
        Object.defineProperties(node, {
          usedBytes: {
            get: function() {
              return this.contents.length;
            }
          }
        });
        // override each stream op with one that tries to force load the lazy
        // file first
        var stream_ops = {};
        for (const [key, fn] of Object.entries(node.stream_ops)) {
          stream_ops[key] = (...args) => {
            FS.forceLoadFile(node);
            return fn(...args);
          };
        }
        function writeChunks(stream, buffer, offset, length, position) {
          var contents = stream.node.contents;
          if (position >= contents.length) return 0;
          var size = Math.min(contents.length - position, length);
          if (contents.slice) {
            // normal array
            for (var i = 0; i < size; i++) {
              buffer[offset + i] = contents[position + i];
            }
          } else {
            for (var i = 0; i < size; i++) {
              // LazyUint8Array from sync binary XHR
              buffer[offset + i] = contents.get(position + i);
            }
          }
          return size;
        }
        // use a custom read function
        stream_ops.read = (stream, buffer, offset, length, position) => {
          FS.forceLoadFile(node);
          return writeChunks(stream, buffer, offset, length, position);
        };
        // use a custom mmap function
        stream_ops.mmap = (stream, length, position, prot, flags) => {
          FS.forceLoadFile(node);
          var ptr = mmapAlloc(length);
          if (!ptr) {
            throw new FS.ErrnoError(48);
          }
          writeChunks(stream, HEAP8, ptr, length, position);
          return {ptr, allocated: true};
        };
        node.stream_ops = stream_ops;
        return node;
      }
    };

    var SYSCALLS = {
      currentUmask: 18,
      calculateAt(dirfd, path, allowEmpty) {
        if (PATH.isAbs(path)) {
          return path;
        }
        // relative path
        var dir;
        if (dirfd === -100) {
          dir = FS.cwd();
        } else {
          var dirstream = SYSCALLS.getStreamFromFD(dirfd);
          dir = dirstream.path;
        }
        if (path.length == 0) {
          if (!allowEmpty) {
            throw new FS.ErrnoError(44);
          }
          return dir;
        }
        return dir + '/' + path;
      },
      writeStat(buf, stat) {
        HEAPU32[((buf) >>> 2) >>> 0] = stat.dev;
        HEAPU32[(((buf) + (4)) >>> 2) >>> 0] = stat.mode;
        HEAPU32[(((buf) + (8)) >>> 2) >>> 0] = stat.nlink;
        HEAPU32[(((buf) + (12)) >>> 2) >>> 0] = stat.uid;
        HEAPU32[(((buf) + (16)) >>> 2) >>> 0] = stat.gid;
        HEAPU32[(((buf) + (20)) >>> 2) >>> 0] = stat.rdev;
        (tempI64 =
             [
               stat.size >>> 0,
               (tempDouble = stat.size,
                (+(Math.abs(tempDouble))) >= 1 ?
                    (tempDouble > 0 ?
                         (+(Math.floor((tempDouble) / 4294967296))) >>> 0 :
                         (~~((+(Math.ceil(
                             (tempDouble - +(((~~(tempDouble))) >>> 0)) /
                             4294967296))))) >>>
                             0) :
                    0)
             ],
         HEAP32[(((buf) + (24)) >>> 2) >>> 0] = tempI64[0],
         HEAP32[(((buf) + (28)) >>> 2) >>> 0] = tempI64[1]);
        HEAP32[(((buf) + (32)) >>> 2) >>> 0] = 4096;
        HEAP32[(((buf) + (36)) >>> 2) >>> 0] = stat.blocks;
        var atime = stat.atime.getTime();
        var mtime = stat.mtime.getTime();
        var ctime = stat.ctime.getTime();
        (tempI64 =
             [
               Math.floor(atime / 1e3) >>> 0,
               (tempDouble = Math.floor(atime / 1e3),
                (+(Math.abs(tempDouble))) >= 1 ?
                    (tempDouble > 0 ?
                         (+(Math.floor((tempDouble) / 4294967296))) >>> 0 :
                         (~~((+(Math.ceil(
                             (tempDouble - +(((~~(tempDouble))) >>> 0)) /
                             4294967296))))) >>>
                             0) :
                    0)
             ],
         HEAP32[(((buf) + (40)) >>> 2) >>> 0] = tempI64[0],
         HEAP32[(((buf) + (44)) >>> 2) >>> 0] = tempI64[1]);
        HEAPU32[(((buf) + (48)) >>> 2) >>> 0] = (atime % 1e3) * 1e3 * 1e3;
        (tempI64 =
             [
               Math.floor(mtime / 1e3) >>> 0,
               (tempDouble = Math.floor(mtime / 1e3),
                (+(Math.abs(tempDouble))) >= 1 ?
                    (tempDouble > 0 ?
                         (+(Math.floor((tempDouble) / 4294967296))) >>> 0 :
                         (~~((+(Math.ceil(
                             (tempDouble - +(((~~(tempDouble))) >>> 0)) /
                             4294967296))))) >>>
                             0) :
                    0)
             ],
         HEAP32[(((buf) + (56)) >>> 2) >>> 0] = tempI64[0],
         HEAP32[(((buf) + (60)) >>> 2) >>> 0] = tempI64[1]);
        HEAPU32[(((buf) + (64)) >>> 2) >>> 0] = (mtime % 1e3) * 1e3 * 1e3;
        (tempI64 =
             [
               Math.floor(ctime / 1e3) >>> 0,
               (tempDouble = Math.floor(ctime / 1e3),
                (+(Math.abs(tempDouble))) >= 1 ?
                    (tempDouble > 0 ?
                         (+(Math.floor((tempDouble) / 4294967296))) >>> 0 :
                         (~~((+(Math.ceil(
                             (tempDouble - +(((~~(tempDouble))) >>> 0)) /
                             4294967296))))) >>>
                             0) :
                    0)
             ],
         HEAP32[(((buf) + (72)) >>> 2) >>> 0] = tempI64[0],
         HEAP32[(((buf) + (76)) >>> 2) >>> 0] = tempI64[1]);
        HEAPU32[(((buf) + (80)) >>> 2) >>> 0] = (ctime % 1e3) * 1e3 * 1e3;
        (tempI64 =
             [
               stat.ino >>> 0,
               (tempDouble = stat.ino,
                (+(Math.abs(tempDouble))) >= 1 ?
                    (tempDouble > 0 ?
                         (+(Math.floor((tempDouble) / 4294967296))) >>> 0 :
                         (~~((+(Math.ceil(
                             (tempDouble - +(((~~(tempDouble))) >>> 0)) /
                             4294967296))))) >>>
                             0) :
                    0)
             ],
         HEAP32[(((buf) + (88)) >>> 2) >>> 0] = tempI64[0],
         HEAP32[(((buf) + (92)) >>> 2) >>> 0] = tempI64[1]);
        return 0;
      },
      writeStatFs(buf, stats) {
        HEAPU32[(((buf) + (4)) >>> 2) >>> 0] = stats.bsize;
        HEAPU32[(((buf) + (60)) >>> 2) >>> 0] = stats.bsize;
        (tempI64 =
             [
               stats.blocks >>> 0,
               (tempDouble = stats.blocks,
                (+(Math.abs(tempDouble))) >= 1 ?
                    (tempDouble > 0 ?
                         (+(Math.floor((tempDouble) / 4294967296))) >>> 0 :
                         (~~((+(Math.ceil(
                             (tempDouble - +(((~~(tempDouble))) >>> 0)) /
                             4294967296))))) >>>
                             0) :
                    0)
             ],
         HEAP32[(((buf) + (8)) >>> 2) >>> 0] = tempI64[0],
         HEAP32[(((buf) + (12)) >>> 2) >>> 0] = tempI64[1]);
        (tempI64 =
             [
               stats.bfree >>> 0,
               (tempDouble = stats.bfree,
                (+(Math.abs(tempDouble))) >= 1 ?
                    (tempDouble > 0 ?
                         (+(Math.floor((tempDouble) / 4294967296))) >>> 0 :
                         (~~((+(Math.ceil(
                             (tempDouble - +(((~~(tempDouble))) >>> 0)) /
                             4294967296))))) >>>
                             0) :
                    0)
             ],
         HEAP32[(((buf) + (16)) >>> 2) >>> 0] = tempI64[0],
         HEAP32[(((buf) + (20)) >>> 2) >>> 0] = tempI64[1]);
        (tempI64 =
             [
               stats.bavail >>> 0,
               (tempDouble = stats.bavail,
                (+(Math.abs(tempDouble))) >= 1 ?
                    (tempDouble > 0 ?
                         (+(Math.floor((tempDouble) / 4294967296))) >>> 0 :
                         (~~((+(Math.ceil(
                             (tempDouble - +(((~~(tempDouble))) >>> 0)) /
                             4294967296))))) >>>
                             0) :
                    0)
             ],
         HEAP32[(((buf) + (24)) >>> 2) >>> 0] = tempI64[0],
         HEAP32[(((buf) + (28)) >>> 2) >>> 0] = tempI64[1]);
        (tempI64 =
             [
               stats.files >>> 0,
               (tempDouble = stats.files,
                (+(Math.abs(tempDouble))) >= 1 ?
                    (tempDouble > 0 ?
                         (+(Math.floor((tempDouble) / 4294967296))) >>> 0 :
                         (~~((+(Math.ceil(
                             (tempDouble - +(((~~(tempDouble))) >>> 0)) /
                             4294967296))))) >>>
                             0) :
                    0)
             ],
         HEAP32[(((buf) + (32)) >>> 2) >>> 0] = tempI64[0],
         HEAP32[(((buf) + (36)) >>> 2) >>> 0] = tempI64[1]);
        (tempI64 =
             [
               stats.ffree >>> 0,
               (tempDouble = stats.ffree,
                (+(Math.abs(tempDouble))) >= 1 ?
                    (tempDouble > 0 ?
                         (+(Math.floor((tempDouble) / 4294967296))) >>> 0 :
                         (~~((+(Math.ceil(
                             (tempDouble - +(((~~(tempDouble))) >>> 0)) /
                             4294967296))))) >>>
                             0) :
                    0)
             ],
         HEAP32[(((buf) + (40)) >>> 2) >>> 0] = tempI64[0],
         HEAP32[(((buf) + (44)) >>> 2) >>> 0] = tempI64[1]);
        HEAPU32[(((buf) + (48)) >>> 2) >>> 0] = stats.fsid;
        HEAPU32[(((buf) + (64)) >>> 2) >>> 0] = stats.flags;
        // ST_NOSUID
        HEAPU32[(((buf) + (56)) >>> 2) >>> 0] = stats.namelen;
      },
      doMsync(addr, stream, len, flags, offset) {
        if (!FS.isFile(stream.node.mode)) {
          throw new FS.ErrnoError(43);
        }
        if (flags & 2) {
          // MAP_PRIVATE calls need not to be synced back to underlying fs
          return 0;
        }
        var buffer = HEAPU8.subarray(addr >>> 0, addr + len >>> 0);
        FS.msync(stream, buffer, offset, len, flags);
      },
      getStreamFromFD(fd) {
        var stream = FS.getStreamChecked(fd);
        return stream;
      },
      varargs: undefined,
      getStr(ptr) {
        var ret = UTF8ToString(ptr);
        return ret;
      }
    };

    function ___syscall_dup(fd) {
      try {
        var old = SYSCALLS.getStreamFromFD(fd);
        return FS.dupStream(old).fd;
      } catch (e) {
        if (typeof FS == 'undefined' || !(e.name === 'ErrnoError')) throw e;
        return -e.errno;
      }
    }

    function ___syscall_faccessat(dirfd, path, amode, flags) {
      path >>>= 0;
      try {
        path = SYSCALLS.getStr(path);
        path = SYSCALLS.calculateAt(dirfd, path);
        if (amode & ~7) {
          // need a valid mode
          return -28;
        }
        var lookup = FS.lookupPath(path, {follow: true});
        var node = lookup.node;
        if (!node) {
          return -44;
        }
        var perms = '';
        if (amode & 4) perms += 'r';
        if (amode & 2) perms += 'w';
        if (amode & 1) perms += 'x';
        if (perms && FS.nodePermissions(node, perms)) {
          return -2;
        }
        return 0;
      } catch (e) {
        if (typeof FS == 'undefined' || !(e.name === 'ErrnoError')) throw e;
        return -e.errno;
      }
    }

    var syscallGetVarargI = () => {
      // the `+` prepended here is necessary to convince the JSCompiler that
      // varargs is indeed a number.
      var ret = HEAP32[((+SYSCALLS.varargs) >>> 2) >>> 0];
      SYSCALLS.varargs += 4;
      return ret;
    };

    var syscallGetVarargP = syscallGetVarargI;

    /** @type {!Int16Array} */ var HEAP16;

    function ___syscall_fcntl64(fd, cmd, varargs) {
      varargs >>>= 0;
      SYSCALLS.varargs = varargs;
      try {
        var stream = SYSCALLS.getStreamFromFD(fd);
        switch (cmd) {
          case 0: {
            var arg = syscallGetVarargI();
            if (arg < 0) {
              return -28;
            }
            while (FS.streams[arg]) {
              arg++;
            }
            var newStream;
            newStream = FS.dupStream(stream, arg);
            return newStream.fd;
          }

          case 1:
          case 2:
            return 0;

            // FD_CLOEXEC makes no sense for a single process.
          case 3:
            return stream.flags;

          case 4: {
            var arg = syscallGetVarargI();
            var mask = 289792;
            stream.flags = (stream.flags & ~mask) | (arg & mask);
            return 0;
          }

          case 12: {
            var arg = syscallGetVarargP();
            var offset = 0;
            // We're always unlocked.
            HEAP16[(((arg) + (offset)) >>> 1) >>> 0] = 2;
            return 0;
          }

          case 13:
          case 14:
            // Pretend that the locking is successful. These are process-level
            // locks, and Emscripten programs are a single process. If we
            // supported linking a filesystem between programs, we'd need to do
            // more here. See
            // https://github.com/emscripten-core/emscripten/issues/23697
            return 0;
        }
        return -28;
      } catch (e) {
        if (typeof FS == 'undefined' || !(e.name === 'ErrnoError')) throw e;
        return -e.errno;
      }
    }

    function ___syscall_fstat64(fd, buf) {
      buf >>>= 0;
      try {
        return SYSCALLS.writeStat(buf, FS.fstat(fd));
      } catch (e) {
        if (typeof FS == 'undefined' || !(e.name === 'ErrnoError')) throw e;
        return -e.errno;
      }
    }

    function ___syscall_ftruncate64(fd, length_low, length_high) {
      var length = convertI32PairToI53Checked(length_low, length_high);
      try {
        if (isNaN(length)) return -22;
        FS.ftruncate(fd, length);
        return 0;
      } catch (e) {
        if (typeof FS == 'undefined' || !(e.name === 'ErrnoError')) throw e;
        return -e.errno;
      }
    }

    var stringToUTF8 = (str, outPtr, maxBytesToWrite) =>
        stringToUTF8Array(str, HEAPU8, outPtr, maxBytesToWrite);

    function ___syscall_getcwd(buf, size) {
      buf >>>= 0;
      size >>>= 0;
      try {
        if (size === 0) return -28;
        var cwd = FS.cwd();
        var cwdLengthInBytes = lengthBytesUTF8(cwd) + 1;
        if (size < cwdLengthInBytes) return -68;
        stringToUTF8(cwd, buf, size);
        return cwdLengthInBytes;
      } catch (e) {
        if (typeof FS == 'undefined' || !(e.name === 'ErrnoError')) throw e;
        return -e.errno;
      }
    }

    function ___syscall_getdents64(fd, dirp, count) {
      dirp >>>= 0;
      count >>>= 0;
      try {
        var stream = SYSCALLS.getStreamFromFD(fd);
        stream.getdents ||= FS.readdir(stream.path);
        var struct_size = 280;
        var pos = 0;
        var off = FS.llseek(stream, 0, 1);
        var startIdx = Math.floor(off / struct_size);
        var endIdx = Math.min(
            stream.getdents.length, startIdx + Math.floor(count / struct_size));
        for (var idx = startIdx; idx < endIdx; idx++) {
          var id;
          var type;
          var name = stream.getdents[idx];
          if (name === '.') {
            id = stream.node.id;
            type = 4;
          } else if (name === '..') {
            var lookup = FS.lookupPath(stream.path, {parent: true});
            id = lookup.node.id;
            type = 4;
          } else {
            var child;
            try {
              child = FS.lookupNode(stream.node, name);
            } catch (e) {
              // If the entry is not a directory, file, or symlink, nodefs
              // lookupNode will raise EINVAL. Skip these and continue.
              if (e?.errno === 28) {
                continue;
              }
              throw e;
            }
            id = child.id;
            type = FS.isChrdev(child.mode) ? 2 : // character device.
        FS.isDir(child.mode) ? 4 : // directory
        FS.isLink(child.mode) ? 10 : // symbolic link.
        8;
          }
          (tempI64 =
               [
                 id >>> 0,
                 (tempDouble = id,
                  (+(Math.abs(tempDouble))) >= 1 ?
                      (tempDouble > 0 ?
                           (+(Math.floor((tempDouble) / 4294967296))) >>> 0 :
                           (~~((+(Math.ceil(
                               (tempDouble - +(((~~(tempDouble))) >>> 0)) /
                               4294967296))))) >>>
                               0) :
                      0)
               ],
           HEAP32[((dirp + pos) >>> 2) >>> 0] = tempI64[0],
           HEAP32[(((dirp + pos) + (4)) >>> 2) >>> 0] = tempI64[1]);
          (tempI64 =
               [
                 (idx + 1) * struct_size >>> 0,
                 (tempDouble = (idx + 1) * struct_size,
                  (+(Math.abs(tempDouble))) >= 1 ?
                      (tempDouble > 0 ?
                           (+(Math.floor((tempDouble) / 4294967296))) >>> 0 :
                           (~~((+(Math.ceil(
                               (tempDouble - +(((~~(tempDouble))) >>> 0)) /
                               4294967296))))) >>>
                               0) :
                      0)
               ],
           HEAP32[(((dirp + pos) + (8)) >>> 2) >>> 0] = tempI64[0],
           HEAP32[(((dirp + pos) + (12)) >>> 2) >>> 0] = tempI64[1]);
          HEAP16[(((dirp + pos) + (16)) >>> 1) >>> 0] = 280;
          HEAP8[(dirp + pos) + (18) >>> 0] = type;
          stringToUTF8(name, dirp + pos + 19, 256);
          pos += struct_size;
        }
        FS.llseek(stream, idx * struct_size, 0);
        return pos;
      } catch (e) {
        if (typeof FS == 'undefined' || !(e.name === 'ErrnoError')) throw e;
        return -e.errno;
      }
    }

    var ___syscall_getuid32 = () => 0;

    function ___syscall_ioctl(fd, op, varargs) {
      varargs >>>= 0;
      SYSCALLS.varargs = varargs;
      try {
        var stream = SYSCALLS.getStreamFromFD(fd);
        switch (op) {
          case 21509: {
            if (!stream.tty) return -59;
            return 0;
          }

          case 21505: {
            if (!stream.tty) return -59;
            if (stream.tty.ops.ioctl_tcgets) {
              var termios = stream.tty.ops.ioctl_tcgets(stream);
              var argp = syscallGetVarargP();
              HEAP32[((argp) >>> 2) >>> 0] = termios.c_iflag || 0;
              HEAP32[(((argp) + (4)) >>> 2) >>> 0] = termios.c_oflag || 0;
              HEAP32[(((argp) + (8)) >>> 2) >>> 0] = termios.c_cflag || 0;
              HEAP32[(((argp) + (12)) >>> 2) >>> 0] = termios.c_lflag || 0;
              for (var i = 0; i < 32; i++) {
                HEAP8[(argp + i) + (17) >>> 0] = termios.c_cc[i] || 0;
              }
              return 0;
            }
            return 0;
          }

          case 21510:
          case 21511:
          case 21512: {
            if (!stream.tty) return -59;
            return 0;
          }

          case 21506:
          case 21507:
          case 21508: {
            if (!stream.tty) return -59;
            if (stream.tty.ops.ioctl_tcsets) {
              var argp = syscallGetVarargP();
              var c_iflag = HEAP32[((argp) >>> 2) >>> 0];
              var c_oflag = HEAP32[(((argp) + (4)) >>> 2) >>> 0];
              var c_cflag = HEAP32[(((argp) + (8)) >>> 2) >>> 0];
              var c_lflag = HEAP32[(((argp) + (12)) >>> 2) >>> 0];
              var c_cc = [];
              for (var i = 0; i < 32; i++) {
                c_cc.push(HEAP8[(argp + i) + (17) >>> 0]);
              }
              return stream.tty.ops.ioctl_tcsets(
                  stream.tty, op, {c_iflag, c_oflag, c_cflag, c_lflag, c_cc});
            }
            return 0;
          }

          case 21519: {
            if (!stream.tty) return -59;
            var argp = syscallGetVarargP();
            HEAP32[((argp) >>> 2) >>> 0] = 0;
            return 0;
          }

          case 21520: {
            if (!stream.tty) return -59;
            return -28;
          }

          case 21537:
          case 21531: {
            var argp = syscallGetVarargP();
            return FS.ioctl(stream, op, argp);
          }

          case 21523: {
            // TODO: in theory we should write to the winsize struct that gets
            // passed in, but for now musl doesn't read anything on it
            if (!stream.tty) return -59;
            if (stream.tty.ops.ioctl_tiocgwinsz) {
              var winsize = stream.tty.ops.ioctl_tiocgwinsz(stream.tty);
              var argp = syscallGetVarargP();
              HEAP16[((argp) >>> 1) >>> 0] = winsize[0];
              HEAP16[(((argp) + (2)) >>> 1) >>> 0] = winsize[1];
            }
            return 0;
          }

          case 21524: {
            // TODO: technically, this ioctl call should change the window size.
            // but, since emscripten doesn't have any concept of a terminal
            // window yet, we'll just silently throw it away as we do TIOCGWINSZ
            if (!stream.tty) return -59;
            return 0;
          }

          case 21515: {
            if (!stream.tty) return -59;
            return 0;
          }

          default:
            return -28;
        }
      } catch (e) {
        if (typeof FS == 'undefined' || !(e.name === 'ErrnoError')) throw e;
        return -e.errno;
      }
    }

    function ___syscall_lstat64(path, buf) {
      path >>>= 0;
      buf >>>= 0;
      try {
        path = SYSCALLS.getStr(path);
        return SYSCALLS.writeStat(buf, FS.lstat(path));
      } catch (e) {
        if (typeof FS == 'undefined' || !(e.name === 'ErrnoError')) throw e;
        return -e.errno;
      }
    }

    function ___syscall_mkdirat(dirfd, path, mode) {
      path >>>= 0;
      try {
        path = SYSCALLS.getStr(path);
        path = SYSCALLS.calculateAt(dirfd, path);
        mode &= ~SYSCALLS.currentUmask;
        FS.mkdir(path, mode, 0);
        return 0;
      } catch (e) {
        if (typeof FS == 'undefined' || !(e.name === 'ErrnoError')) throw e;
        return -e.errno;
      }
    }

    function ___syscall_newfstatat(dirfd, path, buf, flags) {
      path >>>= 0;
      buf >>>= 0;
      try {
        path = SYSCALLS.getStr(path);
        var nofollow = flags & 256;
        var allowEmpty = flags & 4096;
        flags = flags & (~6400);
        path = SYSCALLS.calculateAt(dirfd, path, allowEmpty);
        return SYSCALLS.writeStat(
            buf, nofollow ? FS.lstat(path) : FS.stat(path));
      } catch (e) {
        if (typeof FS == 'undefined' || !(e.name === 'ErrnoError')) throw e;
        return -e.errno;
      }
    }

    function ___syscall_openat(dirfd, path, flags, varargs) {
      path >>>= 0;
      varargs >>>= 0;
      SYSCALLS.varargs = varargs;
      try {
        path = SYSCALLS.getStr(path);
        path = SYSCALLS.calculateAt(dirfd, path);
        var mode = varargs ? syscallGetVarargI() : 0;
        if (flags & 64) {
          mode &= ~SYSCALLS.currentUmask;
        }
        return FS.open(path, flags, mode).fd;
      } catch (e) {
        if (typeof FS == 'undefined' || !(e.name === 'ErrnoError')) throw e;
        return -e.errno;
      }
    }

    function ___syscall_readlinkat(dirfd, path, buf, bufsize) {
      path >>>= 0;
      buf >>>= 0;
      bufsize >>>= 0;
      try {
        path = SYSCALLS.getStr(path);
        path = SYSCALLS.calculateAt(dirfd, path);
        if (bufsize <= 0) return -28;
        var ret = FS.readlink(path);
        var len = Math.min(bufsize, lengthBytesUTF8(ret));
        var endChar = HEAP8[buf + len >>> 0];
        stringToUTF8(ret, buf, bufsize + 1);
        // readlink is one of the rare functions that write out a C string, but
        // does never append a null to the output buffer(!) stringToUTF8()
        // always appends a null byte, so restore the character under the null
        // byte after the write.
        HEAP8[buf + len >>> 0] = endChar;
        return len;
      } catch (e) {
        if (typeof FS == 'undefined' || !(e.name === 'ErrnoError')) throw e;
        return -e.errno;
      }
    }

    function ___syscall_rmdir(path) {
      path >>>= 0;
      try {
        path = SYSCALLS.getStr(path);
        FS.rmdir(path);
        return 0;
      } catch (e) {
        if (typeof FS == 'undefined' || !(e.name === 'ErrnoError')) throw e;
        return -e.errno;
      }
    }

    function ___syscall_stat64(path, buf) {
      path >>>= 0;
      buf >>>= 0;
      try {
        path = SYSCALLS.getStr(path);
        return SYSCALLS.writeStat(buf, FS.stat(path));
      } catch (e) {
        if (typeof FS == 'undefined' || !(e.name === 'ErrnoError')) throw e;
        return -e.errno;
      }
    }

    function ___syscall_unlinkat(dirfd, path, flags) {
      path >>>= 0;
      try {
        path = SYSCALLS.getStr(path);
        path = SYSCALLS.calculateAt(dirfd, path);
        if (!flags) {
          FS.unlink(path);
        } else if (flags === 512) {
          FS.rmdir(path);
        } else {
          return -28;
        }
        return 0;
      } catch (e) {
        if (typeof FS == 'undefined' || !(e.name === 'ErrnoError')) throw e;
        return -e.errno;
      }
    }

    var readI53FromI64 = ptr => HEAPU32[((ptr) >>> 2) >>> 0] +
        HEAP32[(((ptr) + (4)) >>> 2) >>> 0] * 4294967296;

    function ___syscall_utimensat(dirfd, path, times, flags) {
      path >>>= 0;
      times >>>= 0;
      try {
        var nofollow = flags & 256;
        path = SYSCALLS.getStr(path);
        path = SYSCALLS.calculateAt(dirfd, path, true);
        var now = Date.now(), atime, mtime;
        if (!times) {
          atime = now;
          mtime = now;
        } else {
          var seconds = readI53FromI64(times);
          var nanoseconds = HEAP32[(((times) + (8)) >>> 2) >>> 0];
          if (nanoseconds == 1073741823) {
            atime = now;
          } else if (nanoseconds == 1073741822) {
            atime = null;
          } else {
            atime = (seconds * 1e3) + (nanoseconds / (1e3 * 1e3));
          }
          times += 16;
          seconds = readI53FromI64(times);
          nanoseconds = HEAP32[(((times) + (8)) >>> 2) >>> 0];
          if (nanoseconds == 1073741823) {
            mtime = now;
          } else if (nanoseconds == 1073741822) {
            mtime = null;
          } else {
            mtime = (seconds * 1e3) + (nanoseconds / (1e3 * 1e3));
          }
        }
        // null here means UTIME_OMIT was passed. If both were set to UTIME_OMIT
        // then we can skip the call completely.
        if ((mtime ?? atime) !== null) {
          FS.utime(path, atime, mtime, nofollow);
        }
        return 0;
      } catch (e) {
        if (typeof FS == 'undefined' || !(e.name === 'ErrnoError')) throw e;
        return -e.errno;
      }
    }

    var __abort_js = () => abort('');

    function __embind_register_bigint(
        primitiveType, name, size, minRange, maxRange) {
      primitiveType >>>= 0;
      name >>>= 0;
      size >>>= 0;
    }

    var AsciiToString = ptr => {
      ptr >>>= 0;
      var str = '';
      while (1) {
        var ch = HEAPU8[ptr++ >>> 0];
        if (!ch) return str;
        str += String.fromCharCode(ch);
      }
    };

    var awaitingDependencies = {};

    var registeredTypes = {};

    var typeDependencies = {};

    class BindingError extends Error {
      constructor(message) {
        super(message);
        this.name = 'BindingError';
      }
    }

    var throwBindingError = message => {
      throw new BindingError(message);
    };

    /** @param {Object=} options */ function sharedRegisterType(
        rawType, registeredInstance, options = {}) {
      var name = registeredInstance.name;
      if (!rawType) {
        throwBindingError(
            `type "${name}" must have a positive integer typeid pointer`);
      }
      if (registeredTypes.hasOwnProperty(rawType)) {
        if (options.ignoreDuplicateRegistrations) {
          return;
        } else {
          throwBindingError(`Cannot register type '${name}' twice`);
        }
      }
      registeredTypes[rawType] = registeredInstance;
      delete typeDependencies[rawType];
      if (awaitingDependencies.hasOwnProperty(rawType)) {
        var callbacks = awaitingDependencies[rawType];
        delete awaitingDependencies[rawType];
        callbacks.forEach(cb => cb());
      }
    }

    /** @param {Object=} options */ function registerType(
        rawType, registeredInstance, options = {}) {
      return sharedRegisterType(rawType, registeredInstance, options);
    }

    /** @suppress {globalThis} */ function __embind_register_bool(
        rawType, name, trueValue, falseValue) {
      rawType >>>= 0;
      name >>>= 0;
      name = AsciiToString(name);
      registerType(rawType, {
        name,
        fromWireType: function(wt) {
          // ambiguous emscripten ABI: sometimes return values are
          // true or false, and sometimes integers (0 or 1)
          return !!wt;
        },
        toWireType: function(destructors, o) {
          return o ? trueValue : falseValue;
        },
        readValueFromPointer: function(pointer) {
          return this.fromWireType(HEAPU8[pointer >>> 0]);
        },
        destructorFunction: null
      });
    }

    var emval_freelist = [];

    var emval_handles = [0, 1, , 1, null, 1, true, 1, false, 1];

    function __emval_decref(handle) {
      handle >>>= 0;
      if (handle > 9 && 0 === --emval_handles[handle + 1]) {
        var value = emval_handles[handle];
        emval_handles[handle] = undefined;
        emval_freelist.push(handle);
      }
    }

    var Emval = {
      toValue: handle => {
        if (!handle) {
          throwBindingError(`Cannot use deleted val. handle = ${handle}`);
        }
        return emval_handles[handle];
      },
      toHandle: value => {
        switch (value) {
          case undefined:
            return 2;

          case null:
            return 4;

          case true:
            return 6;

          case false:
            return 8;

          default: {
            const handle = emval_freelist.pop() || emval_handles.length;
            emval_handles[handle] = value;
            emval_handles[handle + 1] = 1;
            return handle;
          }
        }
      }
    };

    /** @suppress {globalThis} */ function readPointer(pointer) {
      return this.fromWireType(HEAPU32[((pointer) >>> 2) >>> 0]);
    }

    var EmValType = {
      name: 'emscripten::val',
      fromWireType: handle => {
        var rv = Emval.toValue(handle);
        __emval_decref(handle);
        return rv;
      },
      toWireType: (destructors, value) => Emval.toHandle(value),
      readValueFromPointer: readPointer,
      destructorFunction: null
    };

    function __embind_register_emval(rawType) {
      rawType >>>= 0;
      return registerType(rawType, EmValType);
    }

    /** @type {!Float32Array} */ var HEAPF32;

    /** @type {!Float64Array} */ var HEAPF64;

    var floatReadValueFromPointer = (name, width) => {
      switch (width) {
        case 4:
          return function(pointer) {
            return this.fromWireType(HEAPF32[((pointer) >>> 2) >>> 0]);
          };

        case 8:
          return function(pointer) {
            return this.fromWireType(HEAPF64[((pointer) >>> 3) >>> 0]);
          };

        default:
          throw new TypeError(`invalid float width (${width}): ${name}`);
      }
    };

    var __embind_register_float = function(rawType, name, size) {
      rawType >>>= 0;
      name >>>= 0;
      size >>>= 0;
      name = AsciiToString(name);
      registerType(rawType, {
        name,
        fromWireType: value => value,
        toWireType: (destructors, value) => value,
        readValueFromPointer: floatReadValueFromPointer(name, size),
        destructorFunction: null
      });
    };

    var createNamedFunction = (name, func) =>
        Object.defineProperty(func, 'name', {value: name});

    var runDestructors = destructors => {
      while (destructors.length) {
        var ptr = destructors.pop();
        var del = destructors.pop();
        del(ptr);
      }
    };

    function usesDestructorStack(argTypes) {
      // Skip return value at index 0 - it's not deleted here.
      for (var i = 1; i < argTypes.length; ++i) {
        // The type does not define a destructor function - must use dynamic
        // stack
        if (argTypes[i] !== null &&
            argTypes[i].destructorFunction === undefined) {
          return true;
        }
      }
      return false;
    }

    var runAndAbortIfError = func => {
      try {
        return func();
      } catch (e) {
        abort(e);
      }
    };

    var runtimeKeepalivePush = () => {
      runtimeKeepaliveCounter += 1;
    };

    var runtimeKeepalivePop = () => {
      runtimeKeepaliveCounter -= 1;
    };

    var Asyncify = {
      instrumentWasmImports(imports) {
        var importPattern =
            /^(ReadBufferDataJs|WaitUntilCompletedJs|WaitUntilPipelineCreatedJs|invoke_.*|__asyncjs__.*)$/;
        for (let [x, original] of Object.entries(imports)) {
          if (typeof original == 'function') {
            let isAsyncifyImport = original.isAsync || importPattern.test(x);
          }
        }
      },
      instrumentFunction(original) {
        var wrapper = (...args) => {
          Asyncify.exportCallStack.push(original);
          try {
            return original(...args);
          } finally {
            if (!ABORT) {
              var top = Asyncify.exportCallStack.pop();
              Asyncify.maybeStopUnwind();
            }
          }
        };
        Asyncify.funcWrappers.set(original, wrapper);
        return wrapper;
      },
      instrumentWasmExports(exports) {
        var ret = {};
        for (let [x, original] of Object.entries(exports)) {
          if (typeof original == 'function') {
            var wrapper = Asyncify.instrumentFunction(original);
            ret[x] = wrapper;
          } else {
            ret[x] = original;
          }
        }
        return ret;
      },
      State: {Normal: 0, Unwinding: 1, Rewinding: 2, Disabled: 3},
      state: 0,
      StackSize: 4096,
      currData: null,
      handleSleepReturnValue: 0,
      exportCallStack: [],
      callstackFuncToId: new Map,
      callStackIdToFunc: new Map,
      funcWrappers: new Map,
      callStackId: 0,
      asyncPromiseHandlers: null,
      sleepCallbacks: [],
      getCallStackId(func) {
        if (!Asyncify.callstackFuncToId.has(func)) {
          var id = Asyncify.callStackId++;
          Asyncify.callstackFuncToId.set(func, id);
          Asyncify.callStackIdToFunc.set(id, func);
        }
        return Asyncify.callstackFuncToId.get(func);
      },
      maybeStopUnwind() {
        if (Asyncify.currData && Asyncify.state === Asyncify.State.Unwinding &&
            Asyncify.exportCallStack.length === 0) {
          // We just finished unwinding.
          // Be sure to set the state before calling any other functions to
          // avoid possible infinite recursion here (For example in debug
          // pthread builds the dbg() function itself can call back into
          // WebAssembly to get the current pthread_self() pointer).
          Asyncify.state = Asyncify.State.Normal;
          // Keep the runtime alive so that a re-wind can be done later.
          runAndAbortIfError(_asyncify_stop_unwind);
          if (typeof Fibers != 'undefined') {
            Fibers.trampoline();
          }
        }
      },
      whenDone() {
        return new Promise((resolve, reject) => {
          Asyncify.asyncPromiseHandlers = {resolve, reject};
        });
      },
      allocateData() {
        // An asyncify data structure has three fields:
        //  0  current stack pos
        //  4  max stack pos
        //  8  id of function at bottom of the call stack (callStackIdToFunc[id]
        //  == wasm func)
        // The Asyncify ABI only interprets the first two fields, the rest is
        // for the runtime. We also embed a stack in the same memory region
        // here, right next to the structure. This struct is also defined as
        // asyncify_data_t in emscripten/fiber.h
        var ptr = _malloc(12 + Asyncify.StackSize);
        Asyncify.setDataHeader(ptr, ptr + 12, Asyncify.StackSize);
        Asyncify.setDataRewindFunc(ptr);
        return ptr;
      },
      setDataHeader(ptr, stack, stackSize) {
        HEAPU32[((ptr) >>> 2) >>> 0] = stack;
        HEAPU32[(((ptr) + (4)) >>> 2) >>> 0] = stack + stackSize;
      },
      setDataRewindFunc(ptr) {
        var bottomOfCallStack = Asyncify.exportCallStack[0];
        var rewindId = Asyncify.getCallStackId(bottomOfCallStack);
        HEAP32[(((ptr) + (8)) >>> 2) >>> 0] = rewindId;
      },
      getDataRewindFunc(ptr) {
        var id = HEAP32[(((ptr) + (8)) >>> 2) >>> 0];
        var func = Asyncify.callStackIdToFunc.get(id);
        return func;
      },
      doRewind(ptr) {
        var original = Asyncify.getDataRewindFunc(ptr);
        var func = Asyncify.funcWrappers.get(original);
        // Once we have rewound and the stack we no longer need to artificially
        // keep the runtime alive.
        return callUserCallback(func);
      },
      handleSleep(startAsync) {
        if (ABORT) return;
        if (Asyncify.state === Asyncify.State.Normal) {
          // Prepare to sleep. Call startAsync, and see what happens:
          // if the code decided to call our callback synchronously,
          // then no async operation was in fact begun, and we don't
          // need to do anything.
          var reachedCallback = false;
          var reachedAfterCallback = false;
          startAsync((handleSleepReturnValue = 0) => {
            if (ABORT) return;
            Asyncify.handleSleepReturnValue = handleSleepReturnValue;
            reachedCallback = true;
            if (!reachedAfterCallback) {
              // We are happening synchronously, so no need for async.
              return;
            }
            Asyncify.state = Asyncify.State.Rewinding;
            runAndAbortIfError(() => _asyncify_start_rewind(Asyncify.currData));
            if (typeof MainLoop != 'undefined' && MainLoop.func) {
              MainLoop.resume();
            }
            var asyncWasmReturnValue, isError = false;
            try {
              asyncWasmReturnValue = Asyncify.doRewind(Asyncify.currData);
            } catch (err) {
              asyncWasmReturnValue = err;
              isError = true;
            }
            // Track whether the return value was handled by any promise
            // handlers.
            var handled = false;
            if (!Asyncify.currData) {
              // All asynchronous execution has finished.
              // `asyncWasmReturnValue` now contains the final
              // return value of the exported async WASM function.
              // Note: `asyncWasmReturnValue` is distinct from
              // `Asyncify.handleSleepReturnValue`.
              // `Asyncify.handleSleepReturnValue` contains the return
              // value of the last C function to have executed
              // `Asyncify.handleSleep()`, whereas `asyncWasmReturnValue`
              // contains the return value of the exported WASM function
              // that may have called C functions that
              // call `Asyncify.handleSleep()`.
              var asyncPromiseHandlers = Asyncify.asyncPromiseHandlers;
              if (asyncPromiseHandlers) {
                Asyncify.asyncPromiseHandlers = null;
                (isError ? asyncPromiseHandlers.reject :
                           asyncPromiseHandlers.resolve)(asyncWasmReturnValue);
                handled = true;
              }
            }
            if (isError && !handled) {
              // If there was an error and it was not handled by now, we have no
              // choice but to rethrow that error into the global scope where it
              // can be caught only by `onerror` or
              // `onunhandledpromiserejection`.
              throw asyncWasmReturnValue;
            }
          });
          reachedAfterCallback = true;
          if (!reachedCallback) {
            // A true async operation was begun; start a sleep.
            Asyncify.state = Asyncify.State.Unwinding;
            // TODO: reuse, don't alloc/free every sleep
            Asyncify.currData = Asyncify.allocateData();
            if (typeof MainLoop != 'undefined' && MainLoop.func) {
              MainLoop.pause();
            }
            runAndAbortIfError(() => _asyncify_start_unwind(Asyncify.currData));
          }
        } else if (Asyncify.state === Asyncify.State.Rewinding) {
          // Stop a resume.
          Asyncify.state = Asyncify.State.Normal;
          runAndAbortIfError(_asyncify_stop_rewind);
          _free(Asyncify.currData);
          Asyncify.currData = null;
          // Call all sleep callbacks now that the sleep-resume is all done.
          Asyncify.sleepCallbacks.forEach(callUserCallback);
        } else {
          abort(`invalid state: ${Asyncify.state}`);
        }
        return Asyncify.handleSleepReturnValue;
      },
      handleAsync: startAsync => Asyncify.handleSleep(async wakeUp => {
        // TODO: add error handling as a second param when handleSleep
        // implements it.
        wakeUp(await startAsync());
      })
    };

    function craftInvokerFunction(
        humanName, argTypes, classType, cppInvokerFunc, cppTargetFunc,
        /** boolean= */ isAsync) {
      // humanName: a human-readable string name for the function to be
      // generated. argTypes: An array that contains the embind type objects for
      // all types in the function signature.
      //    argTypes[0] is the type object for the function return value.
      //    argTypes[1] is the type object for function this object/class type,
      //    or null if not crafting an invoker for a class method.
      //    argTypes[2...] are the actual function parameters.
      // classType: The embind type object for the class to be bound, or null if
      // this is not a method of a class. cppInvokerFunc: JS Function object to
      // the C++-side function that interops into C++ code. cppTargetFunc:
      // Function pointer (an integer to FUNCTION_TABLE) to the target C++
      // function the cppInvokerFunc will end up calling. isAsync: Optional. If
      // true, returns an async function. Async bindings are only supported with
      // JSPI.
      var argCount = argTypes.length;
      if (argCount < 2) {
        throwBindingError(
            'argTypes array size mismatch! Must at least get return value and receiver (this) types!');
      }
      var isClassMethodFunc = (argTypes[1] !== null && classType !== null);
      // Free functions with signature "void function()" do not need an invoker
      // that marshalls between wire types.
      // TODO: This omits argument count check - enable only at -O3 or similar.
      //    if (ENABLE_UNSAFE_OPTS && argCount == 2 && argTypes[0].name ==
      //    'void' && !isClassMethodFunc) {
      //       return FUNCTION_TABLE[fn];
      //    }
      // Determine if we need to use a dynamic stack to store the destructors
      // for the function parameters.
      // TODO: Remove this completely once all function invokers are being
      // dynamically generated.
      var needsDestructorStack = usesDestructorStack(argTypes);
      var returns = !argTypes[0].isVoid;
      var expectedArgCount = argCount - 2;
      var argsWired = new Array(expectedArgCount);
      var invokerFuncArgs = [];
      var destructors = [];
      var invokerFn = function(...args) {
        destructors.length = 0;
        var thisWired;
        invokerFuncArgs.length = isClassMethodFunc ? 2 : 1;
        invokerFuncArgs[0] = cppTargetFunc;
        if (isClassMethodFunc) {
          thisWired = argTypes[1].toWireType(destructors, this);
          invokerFuncArgs[1] = thisWired;
        }
        for (var i = 0; i < expectedArgCount; ++i) {
          argsWired[i] = argTypes[i + 2].toWireType(destructors, args[i]);
          invokerFuncArgs.push(argsWired[i]);
        }
        var rv = cppInvokerFunc(...invokerFuncArgs);
        function onDone(rv) {
          if (needsDestructorStack) {
            runDestructors(destructors);
          } else {
            for (var i = isClassMethodFunc ? 1 : 2; i < argTypes.length; i++) {
              var param = i === 1 ? thisWired : argsWired[i - 2];
              if (argTypes[i].destructorFunction !== null) {
                argTypes[i].destructorFunction(param);
              }
            }
          }
          if (returns) {
            return argTypes[0].fromWireType(rv);
          }
        }
        if (Asyncify.currData) {
          return Asyncify.whenDone().then(onDone);
        }
        return onDone(rv);
      };
      return createNamedFunction(humanName, invokerFn);
    }

    var ensureOverloadTable = (proto, methodName, humanName) => {
      if (undefined === proto[methodName].overloadTable) {
        var prevFunc = proto[methodName];
        // Inject an overload resolver function that routes to the appropriate
        // overload based on the number of arguments.
        proto[methodName] = function(...args) {
          // TODO This check can be removed in -O3 level "unsafe" optimizations.
          if (!proto[methodName].overloadTable.hasOwnProperty(args.length)) {
            throwBindingError(`Function '${
                humanName}' called with an invalid number of arguments (${
                args.length}) - expects one of (${
                proto[methodName].overloadTable})!`);
          }
          return proto[methodName].overloadTable[args.length].apply(this, args);
        };
        // Move the previous function into the overload table.
        proto[methodName].overloadTable = [];
        proto[methodName].overloadTable[prevFunc.argCount] = prevFunc;
      }
    };

    /** @param {number=} numArguments */
    var exposePublicSymbol = (name, value, numArguments) => {
      if (Module.hasOwnProperty(name)) {
        if (undefined === numArguments ||
            (undefined !== Module[name].overloadTable &&
             undefined !== Module[name].overloadTable[numArguments])) {
          throwBindingError(`Cannot register public name '${name}' twice`);
        }
        // We are exposing a function with the same name as an existing
        // function. Create an overload table and a function selector that
        // routes between the two.
        ensureOverloadTable(Module, name, name);
        if (Module[name].overloadTable.hasOwnProperty(numArguments)) {
          throwBindingError(
              `Cannot register multiple overloads of a function with the same number of arguments (${
                  numArguments})!`);
        }
        // Add the new function into the overload table.
        Module[name].overloadTable[numArguments] = value;
      } else {
        Module[name] = value;
        Module[name].argCount = numArguments;
      }
    };

    var heap32VectorToArray = (count, firstElement) => {
      var array = [];
      for (var i = 0; i < count; i++) {
        // TODO(https://github.com/emscripten-core/emscripten/issues/17310):
        // Find a way to hoist the `>> 2` or `>> 3` out of this loop.
        array.push(HEAPU32[(((firstElement) + (i * 4)) >>> 2) >>> 0]);
      }
      return array;
    };

    class InternalError extends Error {
      constructor(message) {
        super(message);
        this.name = 'InternalError';
      }
    }

    var throwInternalError = message => {
      throw new InternalError(message);
    };

    /** @param {number=} numArguments */ var replacePublicSymbol =
        (name, value, numArguments) => {
          if (!Module.hasOwnProperty(name)) {
            throwInternalError('Replacing nonexistent public symbol');
          }
          // If there's an overload table for this symbol, replace the symbol in
          // the overload table instead.
          if (undefined !== Module[name].overloadTable &&
              undefined !== numArguments) {
            Module[name].overloadTable[numArguments] = value;
          } else {
            Module[name] = value;
            Module[name].argCount = numArguments;
          }
        };

    var getDynCaller = (sig, ptr, promising = false) => (...args) =>
        dynCall(sig, ptr, args, promising);

    var embind__requireFunction = (signature, rawFunction, isAsync = false) => {
      signature = AsciiToString(signature);
      function makeDynCaller() {
        return getDynCaller(signature, rawFunction);
      }
      var fp = makeDynCaller();
      if (typeof fp != 'function') {
        throwBindingError(`unknown function pointer with signature ${
            signature}: ${rawFunction}`);
      }
      return fp;
    };

    class UnboundTypeError extends Error {}

    var getTypeName = type => {
      var ptr = ___getTypeName(type);
      var rv = AsciiToString(ptr);
      _free(ptr);
      return rv;
    };

    var throwUnboundTypeError = (message, types) => {
      var unboundTypes = [];
      var seen = {};
      function visit(type) {
        if (seen[type]) {
          return;
        }
        if (registeredTypes[type]) {
          return;
        }
        if (typeDependencies[type]) {
          typeDependencies[type].forEach(visit);
          return;
        }
        unboundTypes.push(type);
        seen[type] = true;
      }
      types.forEach(visit);
      throw new UnboundTypeError(
          `${message}: ` + unboundTypes.map(getTypeName).join([', ']));
    };

    var whenDependentTypesAreResolved =
        (myTypes, dependentTypes, getTypeConverters) => {
          myTypes.forEach(type => typeDependencies[type] = dependentTypes);
          function onComplete(typeConverters) {
            var myTypeConverters = getTypeConverters(typeConverters);
            if (myTypeConverters.length !== myTypes.length) {
              throwInternalError('Mismatched type converter count');
            }
            for (var i = 0; i < myTypes.length; ++i) {
              registerType(myTypes[i], myTypeConverters[i]);
            }
          }
          var typeConverters = new Array(dependentTypes.length);
          var unregisteredTypes = [];
          var registered = 0;
          for (let [i, dt] of dependentTypes.entries()) {
            if (registeredTypes.hasOwnProperty(dt)) {
              typeConverters[i] = registeredTypes[dt];
            } else {
              unregisteredTypes.push(dt);
              if (!awaitingDependencies.hasOwnProperty(dt)) {
                awaitingDependencies[dt] = [];
              }
              awaitingDependencies[dt].push(() => {
                typeConverters[i] = registeredTypes[dt];
                ++registered;
                if (registered === unregisteredTypes.length) {
                  onComplete(typeConverters);
                }
              });
            }
          }
          if (0 === unregisteredTypes.length) {
            onComplete(typeConverters);
          }
        };

    var getFunctionName = signature => {
      signature = signature.trim();
      const argsIndex = signature.indexOf('(');
      if (argsIndex === -1) return signature;
      return signature.slice(0, argsIndex);
    };

    function __embind_register_function(
        name, argCount, rawArgTypesAddr, signature, rawInvoker, fn, isAsync,
        isNonnullReturn) {
      name >>>= 0;
      rawArgTypesAddr >>>= 0;
      signature >>>= 0;
      rawInvoker >>>= 0;
      fn >>>= 0;
      var argTypes = heap32VectorToArray(argCount, rawArgTypesAddr);
      name = AsciiToString(name);
      name = getFunctionName(name);
      rawInvoker = embind__requireFunction(signature, rawInvoker, isAsync);
      exposePublicSymbol(name, function() {
        throwUnboundTypeError(
            `Cannot call ${name} due to unbound types`, argTypes);
      }, argCount - 1);
      whenDependentTypesAreResolved([], argTypes, argTypes => {
        var invokerArgsArray = [argTypes[0], null].concat(argTypes.slice(1));
        replacePublicSymbol(
            name,
            craftInvokerFunction(
                name, invokerArgsArray, null, rawInvoker, fn, isAsync),
            argCount - 1);
        return [];
      });
    }

    /** @type {!Uint16Array} */ var HEAPU16;

    var integerReadValueFromPointer = (name, width, signed) => {
      // integers are quite common, so generate very specialized functions
      switch (width) {
        case 1:
          return signed ? pointer => HEAP8[pointer >>> 0] :
                          pointer => HEAPU8[pointer >>> 0];

        case 2:
          return signed ? pointer => HEAP16[((pointer) >>> 1) >>> 0] :
                          pointer => HEAPU16[((pointer) >>> 1) >>> 0];

        case 4:
          return signed ? pointer => HEAP32[((pointer) >>> 2) >>> 0] :
                          pointer => HEAPU32[((pointer) >>> 2) >>> 0];

        default:
          throw new TypeError(`invalid integer width (${width}): ${name}`);
      }
    };

    /** @suppress {globalThis} */ var __embind_register_integer = function(
        primitiveType, name, size, minRange, maxRange) {
      primitiveType >>>= 0;
      name >>>= 0;
      size >>>= 0;
      name = AsciiToString(name);
      const isUnsignedType = minRange === 0;
      let fromWireType = value => value;
      if (isUnsignedType) {
        var bitshift = 32 - 8 * size;
        fromWireType = value => (value << bitshift) >>> bitshift;
        maxRange = fromWireType(maxRange);
      }
      registerType(primitiveType, {
        name,
        fromWireType,
        toWireType: (destructors, value) => value,
        readValueFromPointer:
            integerReadValueFromPointer(name, size, minRange !== 0),
        destructorFunction: null
      });
    };

    function __embind_register_memory_view(rawType, dataTypeIndex, name) {
      rawType >>>= 0;
      name >>>= 0;
      var typeMapping = [
        Int8Array, Uint8Array, Int16Array, Uint16Array, Int32Array, Uint32Array,
        Float32Array, Float64Array
      ];
      var TA = typeMapping[dataTypeIndex];
      function decodeMemoryView(handle) {
        var size = HEAPU32[((handle) >>> 2) >>> 0];
        var data = HEAPU32[(((handle) + (4)) >>> 2) >>> 0];
        return new TA(HEAP8.buffer, data, size);
      }
      name = AsciiToString(name);
      registerType(
          rawType, {
            name,
            fromWireType: decodeMemoryView,
            readValueFromPointer: decodeMemoryView
          },
          {ignoreDuplicateRegistrations: true});
    }

    function __embind_register_std_string(rawType, name) {
      rawType >>>= 0;
      name >>>= 0;
      name = AsciiToString(name);
      var stdStringIsUTF8 = true;
      registerType(rawType, {
        name,
        // For some method names we use string keys here since they are part of
        // the public/external API and/or used by the runtime-generated code.
        fromWireType(value) {
          var length = HEAPU32[((value) >>> 2) >>> 0];
          var payload = value + 4;
          var str;
          if (stdStringIsUTF8) {
            str = UTF8ToString(payload, length, true);
          } else {
            str = '';
            for (var i = 0; i < length; ++i) {
              str += String.fromCharCode(HEAPU8[payload + i >>> 0]);
            }
          }
          _free(value);
          return str;
        },
        toWireType(destructors, value) {
          if (value instanceof ArrayBuffer) {
            value = new Uint8Array(value);
          }
          var length;
          var valueIsOfTypeString = (typeof value == 'string');
          // We accept `string` or array views with single byte elements
          if (!(valueIsOfTypeString ||
                (ArrayBuffer.isView(value) && value.BYTES_PER_ELEMENT == 1))) {
            throwBindingError('Cannot pass non-string to std::string');
          }
          if (stdStringIsUTF8 && valueIsOfTypeString) {
            length = lengthBytesUTF8(value);
          } else {
            length = value.length;
          }
          // assumes POINTER_SIZE alignment
          var base = _malloc(4 + length + 1);
          var ptr = base + 4;
          HEAPU32[((base) >>> 2) >>> 0] = length;
          if (valueIsOfTypeString) {
            if (stdStringIsUTF8) {
              stringToUTF8(value, ptr, length + 1);
            } else {
              for (var i = 0; i < length; ++i) {
                var charCode = value.charCodeAt(i);
                if (charCode > 255) {
                  _free(base);
                  throwBindingError(
                      'String has UTF-16 code units that do not fit in 8 bits');
                }
                HEAPU8[ptr + i >>> 0] = charCode;
              }
            }
          } else {
            HEAPU8.set(value, ptr >>> 0);
          }
          if (destructors !== null) {
            destructors.push(_free, base);
          }
          return base;
        },
        readValueFromPointer: readPointer,
        destructorFunction(ptr) {
          _free(ptr);
        }
      });
    }

    var UTF16Decoder = new TextDecoder('utf-16le');

    var UTF16ToString = (ptr, maxBytesToRead, ignoreNul) => {
      var idx = ((ptr) >>> 1);
      var endIdx = findStringEnd(HEAPU16, idx, maxBytesToRead / 2, ignoreNul);
      return UTF16Decoder.decode(HEAPU16.subarray(idx >>> 0, endIdx >>> 0));
    };

    var stringToUTF16 = (str, outPtr, maxBytesToWrite = 2147483647) => {
      if (maxBytesToWrite < 2) return 0;
      maxBytesToWrite -= 2;
      // Null terminator.
      var startPtr = outPtr;
      var numCharsToWrite = (maxBytesToWrite < str.length * 2) ?
          (maxBytesToWrite / 2) :
          str.length;
      for (var i = 0; i < numCharsToWrite; ++i) {
        // charCodeAt returns a UTF-16 encoded code unit, so it can be directly
        // written to the HEAP.
        var codeUnit = str.charCodeAt(i);
        // possibly a lead surrogate
        HEAP16[((outPtr) >>> 1) >>> 0] = codeUnit;
        outPtr += 2;
      }
      // Null-terminate the pointer to the HEAP.
      HEAP16[((outPtr) >>> 1) >>> 0] = 0;
      return outPtr - startPtr;
    };

    var lengthBytesUTF16 = str => str.length * 2;

    var UTF32ToString = (ptr, maxBytesToRead, ignoreNul) => {
      var str = '';
      var startIdx = ((ptr) >>> 2);
      // If maxBytesToRead is not passed explicitly, it will be undefined, and
      // this will always evaluate to true. This saves on code size.
      for (var i = 0; !(i >= maxBytesToRead / 4); i++) {
        var utf32 = HEAPU32[startIdx + i >>> 0];
        if (!utf32 && !ignoreNul) break;
        str += String.fromCodePoint(utf32);
      }
      return str;
    };

    var stringToUTF32 = (str, outPtr, maxBytesToWrite = 2147483647) => {
      outPtr >>>= 0;
      if (maxBytesToWrite < 4) return 0;
      var startPtr = outPtr;
      var endPtr = startPtr + maxBytesToWrite - 4;
      for (var i = 0; i < str.length; ++i) {
        var codePoint = str.codePointAt(i);
        // Gotcha: if codePoint is over 0xFFFF, it is represented as a surrogate
        // pair in UTF-16. We need to manually skip over the second code unit
        // for correct iteration.
        if (codePoint > 65535) {
          i++;
        }
        HEAP32[((outPtr) >>> 2) >>> 0] = codePoint;
        outPtr += 4;
        if (outPtr + 4 > endPtr) break;
      }
      // Null-terminate the pointer to the HEAP.
      HEAP32[((outPtr) >>> 2) >>> 0] = 0;
      return outPtr - startPtr;
    };

    var lengthBytesUTF32 = str => {
      var len = 0;
      for (var i = 0; i < str.length; ++i) {
        var codePoint = str.codePointAt(i);
        // Gotcha: if codePoint is over 0xFFFF, it is represented as a surrogate
        // pair in UTF-16. We need to manually skip over the second code unit
        // for correct iteration.
        if (codePoint > 65535) {
          i++;
        }
        len += 4;
      }
      return len;
    };

    function __embind_register_std_wstring(rawType, charSize, name) {
      rawType >>>= 0;
      charSize >>>= 0;
      name >>>= 0;
      name = AsciiToString(name);
      var decodeString, encodeString, lengthBytesUTF;
      if (charSize === 2) {
        decodeString = UTF16ToString;
        encodeString = stringToUTF16;
        lengthBytesUTF = lengthBytesUTF16;
      } else {
        decodeString = UTF32ToString;
        encodeString = stringToUTF32;
        lengthBytesUTF = lengthBytesUTF32;
      }
      registerType(rawType, {
        name,
        fromWireType: value => {
          // Code mostly taken from _embind_register_std_string fromWireType
          var length = HEAPU32[((value) >>> 2) >>> 0];
          var str = decodeString(value + 4, length * charSize, true);
          _free(value);
          return str;
        },
        toWireType: (destructors, value) => {
          if (!(typeof value == 'string')) {
            throwBindingError(
                `Cannot pass non-string to C++ string type ${name}`);
          }
          // assumes POINTER_SIZE alignment
          var length = lengthBytesUTF(value);
          var ptr = _malloc(4 + length + charSize);
          HEAPU32[((ptr) >>> 2) >>> 0] = length / charSize;
          encodeString(value, ptr + 4, length + charSize);
          if (destructors !== null) {
            destructors.push(_free, ptr);
          }
          return ptr;
        },
        readValueFromPointer: readPointer,
        destructorFunction(ptr) {
          _free(ptr);
        }
      });
    }

    var __embind_register_void = function(rawType, name) {
      rawType >>>= 0;
      name >>>= 0;
      name = AsciiToString(name);
      registerType(rawType, {
        isVoid: true,
        // void return values can be optimized out sometimes
        name,
        fromWireType: () => undefined,
        // TODO: assert if anything else is given?
        toWireType: (destructors, o) => undefined
      });
    };

    var emval_methodCallers = [];

    var emval_addMethodCaller = caller => {
      var id = emval_methodCallers.length;
      emval_methodCallers.push(caller);
      return id;
    };

    var requireRegisteredType = (rawType, humanName) => {
      var impl = registeredTypes[rawType];
      if (undefined === impl) {
        throwBindingError(
            `${humanName} has unknown type ${getTypeName(rawType)}`);
      }
      return impl;
    };

    var emval_lookupTypes = (argCount, argTypes) => {
      var a = new Array(argCount);
      for (var i = 0; i < argCount; ++i) {
        a[i] = requireRegisteredType(
            HEAPU32[(((argTypes) + (i * 4)) >>> 2) >>> 0], `parameter ${i}`);
      }
      return a;
    };

    var emval_returnValue = (toReturnWire, destructorsRef, handle) => {
      var destructors = [];
      var result = toReturnWire(destructors, handle);
      if (destructors.length) {
        // void, primitives and any other types w/o destructors don't need to
        // allocate a handle
        HEAPU32[((destructorsRef) >>> 2) >>> 0] = Emval.toHandle(destructors);
      }
      return result;
    };

    var emval_symbols = {};

    var getStringOrSymbol = address => {
      var symbol = emval_symbols[address];
      if (symbol === undefined) {
        return AsciiToString(address);
      }
      return symbol;
    };

    var __emval_create_invoker = function(argCount, argTypesPtr, kind) {
      argTypesPtr >>>= 0;
      var GenericWireTypeSize = 8;
      var [retType, ...argTypes] = emval_lookupTypes(argCount, argTypesPtr);
      var toReturnWire = retType.toWireType.bind(retType);
      var argFromPtr =
          argTypes.map(type => type.readValueFromPointer.bind(type));
      argCount--;
      // remove the extracted return type
      var argN = new Array(argCount);
      var invokerFunction = (handle, methodName, destructorsRef, args) => {
        var offset = 0;
        for (var i = 0; i < argCount; ++i) {
          argN[i] = argFromPtr[i](args + offset);
          offset += GenericWireTypeSize;
        }
        var rv;
        switch (kind) {
          case 0:
            rv = Emval.toValue(handle).apply(null, argN);
            break;

          case 2:
            rv = Reflect.construct(Emval.toValue(handle), argN);
            break;

          case 3:
            // no-op, just return the argument
            rv = argN[0];
            break;

          case 1:
            rv = Emval.toValue(handle)[getStringOrSymbol(methodName)](...argN);
            break;
        }
        return emval_returnValue(toReturnWire, destructorsRef, rv);
      };
      var functionName =
          `methodCaller<(${argTypes.map(t => t.name)}) => ${retType.name}>`;
      return emval_addMethodCaller(
          createNamedFunction(functionName, invokerFunction));
    };

    function __emval_incref(handle) {
      handle >>>= 0;
      if (handle > 9) {
        emval_handles[handle + 1] += 1;
      }
    }

    function __emval_invoke(caller, handle, methodName, destructorsRef, args) {
      caller >>>= 0;
      handle >>>= 0;
      methodName >>>= 0;
      destructorsRef >>>= 0;
      args >>>= 0;
      return emval_methodCallers[caller](
          handle, methodName, destructorsRef, args);
    }

    function __emval_run_destructors(handle) {
      handle >>>= 0;
      var destructors = Emval.toValue(handle);
      runDestructors(destructors);
      __emval_decref(handle);
    }

    function __gmtime_js(time_low, time_high, tmPtr) {
      var time = convertI32PairToI53Checked(time_low, time_high);
      tmPtr >>>= 0;
      var date = new Date(time * 1e3);
      if (isNaN(date.getTime())) {
        return 1;
      }
      HEAP32[((tmPtr) >>> 2) >>> 0] = date.getUTCSeconds();
      HEAP32[(((tmPtr) + (4)) >>> 2) >>> 0] = date.getUTCMinutes();
      HEAP32[(((tmPtr) + (8)) >>> 2) >>> 0] = date.getUTCHours();
      HEAP32[(((tmPtr) + (12)) >>> 2) >>> 0] = date.getUTCDate();
      HEAP32[(((tmPtr) + (16)) >>> 2) >>> 0] = date.getUTCMonth();
      HEAP32[(((tmPtr) + (20)) >>> 2) >>> 0] = date.getUTCFullYear() - 1900;
      HEAP32[(((tmPtr) + (24)) >>> 2) >>> 0] = date.getUTCDay();
      var start = Date.UTC(date.getUTCFullYear(), 0, 1, 0, 0, 0, 0);
      var yday = ((date.getTime() - start) / (1e3 * 60 * 60 * 24)) | 0;
      HEAP32[(((tmPtr) + (28)) >>> 2) >>> 0] = yday;
      return 0;
    }

    var isLeapYear = year =>
        year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);

    var MONTH_DAYS_LEAP_CUMULATIVE =
        [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335];

    var MONTH_DAYS_REGULAR_CUMULATIVE =
        [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];

    var ydayFromDate = date => {
      var leap = isLeapYear(date.getFullYear());
      var monthDaysCumulative =
          (leap ? MONTH_DAYS_LEAP_CUMULATIVE : MONTH_DAYS_REGULAR_CUMULATIVE);
      var yday = monthDaysCumulative[date.getMonth()] + date.getDate() - 1;
      // -1 since it's days since Jan 1
      return yday;
    };

    function __localtime_js(time_low, time_high, tmPtr) {
      var time = convertI32PairToI53Checked(time_low, time_high);
      tmPtr >>>= 0;
      var date = new Date(time * 1e3);
      if (isNaN(date.getTime())) {
        return 1;
      }
      HEAP32[((tmPtr) >>> 2) >>> 0] = date.getSeconds();
      HEAP32[(((tmPtr) + (4)) >>> 2) >>> 0] = date.getMinutes();
      HEAP32[(((tmPtr) + (8)) >>> 2) >>> 0] = date.getHours();
      HEAP32[(((tmPtr) + (12)) >>> 2) >>> 0] = date.getDate();
      HEAP32[(((tmPtr) + (16)) >>> 2) >>> 0] = date.getMonth();
      HEAP32[(((tmPtr) + (20)) >>> 2) >>> 0] = date.getFullYear() - 1900;
      HEAP32[(((tmPtr) + (24)) >>> 2) >>> 0] = date.getDay();
      var yday = ydayFromDate(date) | 0;
      HEAP32[(((tmPtr) + (28)) >>> 2) >>> 0] = yday;
      HEAP32[(((tmPtr) + (36)) >>> 2) >>> 0] = -(date.getTimezoneOffset() * 60);
      // Attention: DST is in December in South, and some regions don't have DST
      // at all.
      var start = new Date(date.getFullYear(), 0, 1);
      var summerOffset = new Date(date.getFullYear(), 6, 1).getTimezoneOffset();
      var winterOffset = start.getTimezoneOffset();
      var dst =
          (summerOffset != winterOffset &&
           date.getTimezoneOffset() == Math.min(winterOffset, summerOffset)) |
          0;
      HEAP32[(((tmPtr) + (32)) >>> 2) >>> 0] = dst;
      return 0;
    }

    var setTempRet0 = val => __emscripten_tempret_set(val);

    var __mktime_js = function(tmPtr) {
      tmPtr >>>= 0;
      var ret = (() => {
        var date = new Date(
            HEAP32[(((tmPtr) + (20)) >>> 2) >>> 0] + 1900,
            HEAP32[(((tmPtr) + (16)) >>> 2) >>> 0],
            HEAP32[(((tmPtr) + (12)) >>> 2) >>> 0],
            HEAP32[(((tmPtr) + (8)) >>> 2) >>> 0],
            HEAP32[(((tmPtr) + (4)) >>> 2) >>> 0],
            HEAP32[((tmPtr) >>> 2) >>> 0], 0);
        if (isNaN(date.getTime())) {
          return -1;
        }
        // There's an ambiguous hour when the time goes back; the tm_isdst field
        // is used to disambiguate it.  Date() basically guesses, so we fix it
        // up if it guessed wrong, or fill in tm_isdst with the guess if it's
        // -1.
        var dst = HEAP32[(((tmPtr) + (32)) >>> 2) >>> 0];
        var guessedOffset = date.getTimezoneOffset();
        var start = new Date(date.getFullYear(), 0, 1);
        var summerOffset =
            new Date(date.getFullYear(), 6, 1).getTimezoneOffset();
        var winterOffset = start.getTimezoneOffset();
        var dstOffset = Math.min(winterOffset, summerOffset);
        // DST is in December in South
        if (dst < 0) {
          // Attention: some regions don't have DST at all.
          dst = Number(
              summerOffset != winterOffset && dstOffset == guessedOffset);
        } else if ((dst > 0) != (dstOffset == guessedOffset)) {
          var nonDstOffset = Math.max(winterOffset, summerOffset);
          var trueOffset = dst > 0 ? dstOffset : nonDstOffset;
          // Don't try setMinutes(date.getMinutes() + ...) -- it's messed up.
          date.setTime(date.getTime() + (trueOffset - guessedOffset) * 6e4);
          if (isNaN(date.getTime())) {
            return -1;
          }
        }
        HEAP32[(((tmPtr) + (32)) >>> 2) >>> 0] = dst;
        HEAP32[(((tmPtr) + (24)) >>> 2) >>> 0] = date.getDay();
        var yday = ydayFromDate(date) | 0;
        HEAP32[(((tmPtr) + (28)) >>> 2) >>> 0] = yday;
        // To match expected behavior, update fields from date
        HEAP32[((tmPtr) >>> 2) >>> 0] = date.getSeconds();
        HEAP32[(((tmPtr) + (4)) >>> 2) >>> 0] = date.getMinutes();
        HEAP32[(((tmPtr) + (8)) >>> 2) >>> 0] = date.getHours();
        HEAP32[(((tmPtr) + (12)) >>> 2) >>> 0] = date.getDate();
        HEAP32[(((tmPtr) + (16)) >>> 2) >>> 0] = date.getMonth();
        HEAP32[(((tmPtr) + (20)) >>> 2) >>> 0] = date.getYear();
        // Return time in seconds
        return date.getTime() / 1e3;
      })();
      return (
          setTempRet0(
              (tempDouble = ret,
               (+(Math.abs(tempDouble))) >= 1 ?
                   (tempDouble > 0 ?
                        (+(Math.floor((tempDouble) / 4294967296))) >>> 0 :
                        (~~((+(Math.ceil(
                            (tempDouble - +(((~~(tempDouble))) >>> 0)) /
                            4294967296))))) >>>
                            0) :
                   0)),
          ret >>> 0);
    };

    function __mmap_js(
        len, prot, flags, fd, offset_low, offset_high, allocated, addr) {
      len >>>= 0;
      var offset = convertI32PairToI53Checked(offset_low, offset_high);
      allocated >>>= 0;
      addr >>>= 0;
      try {
        var stream = SYSCALLS.getStreamFromFD(fd);
        var res = FS.mmap(stream, len, offset, prot, flags);
        var ptr = res.ptr;
        HEAP32[((allocated) >>> 2) >>> 0] = res.allocated;
        HEAPU32[((addr) >>> 2) >>> 0] = ptr;
        return 0;
      } catch (e) {
        if (typeof FS == 'undefined' || !(e.name === 'ErrnoError')) throw e;
        return -e.errno;
      }
    }

    function __munmap_js(addr, len, prot, flags, fd, offset_low, offset_high) {
      addr >>>= 0;
      len >>>= 0;
      var offset = convertI32PairToI53Checked(offset_low, offset_high);
      try {
        var stream = SYSCALLS.getStreamFromFD(fd);
        if (prot & 2) {
          SYSCALLS.doMsync(addr, stream, len, flags, offset);
        }
      } catch (e) {
        if (typeof FS == 'undefined' || !(e.name === 'ErrnoError')) throw e;
        return -e.errno;
      }
    }

    var __tzset_js = function(timezone, daylight, std_name, dst_name) {
      timezone >>>= 0;
      daylight >>>= 0;
      std_name >>>= 0;
      dst_name >>>= 0;
      // TODO: Use (malleable) environment variables instead of system settings.
      var currentYear = (new Date).getFullYear();
      var winter = new Date(currentYear, 0, 1);
      var summer = new Date(currentYear, 6, 1);
      var winterOffset = winter.getTimezoneOffset();
      var summerOffset = summer.getTimezoneOffset();
      // Local standard timezone offset. Local standard time is not adjusted for
      // daylight savings.  This code uses the fact that getTimezoneOffset
      // returns a greater value during Standard Time versus Daylight Saving
      // Time (DST). Thus it determines the expected output during Standard
      // Time, and it compares whether the output of the given date the same
      // (Standard) or less (DST).
      var stdTimezoneOffset = Math.max(winterOffset, summerOffset);
      // timezone is specified as seconds west of UTC ("The external variable
      // `timezone` shall be set to the difference, in seconds, between
      // Coordinated Universal Time (UTC) and local standard time."), the same
      // as returned by stdTimezoneOffset.
      // See http://pubs.opengroup.org/onlinepubs/009695399/functions/tzset.html
      HEAPU32[((timezone) >>> 2) >>> 0] = stdTimezoneOffset * 60;
      HEAP32[((daylight) >>> 2) >>> 0] = Number(winterOffset != summerOffset);
      var extractZone = timezoneOffset => {
        // Why inverse sign?
        // Read here
        // https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Date/getTimezoneOffset
        var sign = timezoneOffset >= 0 ? '-' : '+';
        var absOffset = Math.abs(timezoneOffset);
        var hours = String(Math.floor(absOffset / 60)).padStart(2, '0');
        var minutes = String(absOffset % 60).padStart(2, '0');
        return `UTC${sign}${hours}${minutes}`;
      };
      var winterName = extractZone(winterOffset);
      var summerName = extractZone(summerOffset);
      if (summerOffset < winterOffset) {
        // Northern hemisphere
        stringToUTF8(winterName, std_name, 17);
        stringToUTF8(summerName, dst_name, 17);
      } else {
        stringToUTF8(winterName, dst_name, 17);
        stringToUTF8(summerName, std_name, 17);
      }
    };

    var _emscripten_get_now = () => performance.now();

    var _emscripten_date_now = () => Date.now();

    var nowIsMonotonic = 1;

    var checkWasiClock = clock_id => clock_id >= 0 && clock_id <= 3;

    function _clock_time_get(
        clk_id, ignored_precision_low, ignored_precision_high, ptime) {
      var ignored_precision = convertI32PairToI53Checked(
          ignored_precision_low, ignored_precision_high);
      ptime >>>= 0;
      if (!checkWasiClock(clk_id)) {
        return 28;
      }
      var now;
      // all wasi clocks but realtime are monotonic
      if (clk_id === 0) {
        now = _emscripten_date_now();
      } else if (nowIsMonotonic) {
        now = _emscripten_get_now();
      } else {
        return 52;
      }
      // "now" is in ms, and wasi times are in ns.
      var nsec = Math.round(now * 1e3 * 1e3);
      (tempI64 =
           [
             nsec >>> 0,
             (tempDouble = nsec,
              (+(Math.abs(tempDouble))) >= 1 ?
                  (tempDouble > 0 ?
                       (+(Math.floor((tempDouble) / 4294967296))) >>> 0 :
                       (~~((+(Math.ceil(
                           (tempDouble - +(((~~(tempDouble))) >>> 0)) /
                           4294967296))))) >>>
                           0) :
                  0)
           ],
       HEAP32[((ptime) >>> 2) >>> 0] = tempI64[0],
       HEAP32[(((ptime) + (4)) >>> 2) >>> 0] = tempI64[1]);
      return 0;
    }

    var readEmAsmArgsArray = [];

    var readEmAsmArgs = (sigPtr, buf) => {
      readEmAsmArgsArray.length = 0;
      var ch;
      // Most arguments are i32s, so shift the buffer pointer so it is a plain
      // index into HEAP32.
      while (ch = HEAPU8[sigPtr++ >>> 0]) {
        // Floats are always passed as doubles, so all types except for 'i'
        // are 8 bytes and require alignment.
        var wide = (ch != 105);
        wide &= (ch != 112);
        buf += wide && (buf % 8) ? 4 : 0;
        readEmAsmArgsArray.push(// Special case for pointers under wasm64 or CAN_ADDRESS_2GB mode.
    ch == 112 ? HEAPU32[((buf) >>> 2) >>> 0] : ch == 105 ? HEAP32[((buf) >>> 2) >>> 0] : HEAPF64[((buf) >>> 3) >>> 0]);
        buf += wide ? 8 : 4;
      }
      return readEmAsmArgsArray;
    };

    var runEmAsmFunction = (code, sigPtr, argbuf) => {
      var args = readEmAsmArgs(sigPtr, argbuf);
      return ASM_CONSTS[code](...args);
    };

    function _emscripten_asm_const_int(code, sigPtr, argbuf) {
      code >>>= 0;
      sigPtr >>>= 0;
      argbuf >>>= 0;
      return runEmAsmFunction(code, sigPtr, argbuf);
    }

    function _emscripten_errn(str, len) {
      str >>>= 0;
      len >>>= 0;
      return err(UTF8ToString(str, len));
    }

    var getHeapMax = () =>  // Stay one Wasm page short of 4GB: while e.g.
                            // Chrome is able to allocate
        // full 4GB Wasm memories, the size will wrap back to 0 bytes in Wasm
        // side for any code that deals with heap sizes, which would require
        // special casing all heap size related code to treat 0 specially.
        4294901760;

    function _emscripten_get_heap_max() {
      return getHeapMax();
    }

    var _emscripten_has_asyncify = () => 1;

    function _emscripten_outn(str, len) {
      str >>>= 0;
      len >>>= 0;
      return out(UTF8ToString(str, len));
    }

    var UNWIND_CACHE = {};

    var stringToNewUTF8 = str => {
      var size = lengthBytesUTF8(str) + 1;
      var ret = _malloc(size);
      if (ret) stringToUTF8(str, ret, size);
      return ret;
    };

    /** @returns {number} */ var convertFrameToPC = frame => {
      var match;
      if (match = /\bwasm-function\[\d+\]:(0x[0-9a-f]+)/.exec(frame)) {
        // Wasm engines give the binary offset directly, so we use that as
        // return address
        return +match[1];
      } else if (match = /:(\d+):\d+(?:\)|$)/.exec(frame)) {
        // If we are in js, we can use the js line number as the "return
        // address". This should work for wasm2js.  We tag the high bit to
        // distinguish this from wasm addresses.
        return 2147483648 | +match[1];
      }
      // return 0 if we can't find any
      return 0;
    };

    var saveInUnwindCache = callstack => {
      for (var line of callstack) {
        var pc = convertFrameToPC(line);
        if (pc) {
          UNWIND_CACHE[pc] = line;
        }
      }
    };

    var jsStackTrace = () => (new Error).stack.toString();

    function _emscripten_stack_snapshot() {
      var callstack = jsStackTrace().split('\n');
      if (callstack[0] == 'Error') {
        callstack.shift();
      }
      saveInUnwindCache(callstack);
      // Caches the stack snapshot so that emscripten_stack_unwind_buffer() can
      // unwind from this spot.
      UNWIND_CACHE.last_addr = convertFrameToPC(callstack[3]);
      UNWIND_CACHE.last_stack = callstack;
      return UNWIND_CACHE.last_addr;
    }

    function _emscripten_pc_get_function(pc) {
      pc >>>= 0;
      var frame = UNWIND_CACHE[pc];
      if (!frame) return 0;
      var name;
      var match;
      // First try to match foo.wasm.sym files explcitly. e.g.
      //   at test_return_address.wasm.main
      //   (wasm://wasm/test_return_address.wasm-0012cc2a:wasm-function[26]:0x9f3
      // Then match JS symbols which don't include that module name:
      //   at invokeEntryPoint (.../test_return_address.js:1500:42)
      // Finally match firefox format:
      //   Object._main@http://server.com:4324:12'
      if (match = /^\s+at .*\.wasm\.(.*) \(.*\)$/.exec(frame)) {
        name = match[1];
      } else if (match = /^\s+at (.*) \(.*\)$/.exec(frame)) {
        name = match[1];
      } else if (match = /^(.+?)@/.exec(frame)) {
        name = match[1];
      } else {
        return 0;
      }
      _free(_emscripten_pc_get_function.ret ?? 0);
      _emscripten_pc_get_function.ret = stringToNewUTF8(name);
      return _emscripten_pc_get_function.ret;
    }

    var growMemory = size => {
      var oldHeapSize = wasmMemory.buffer.byteLength;
      var pages = ((size - oldHeapSize + 65535) / 65536) | 0;
      try {
        // round size grow request up to wasm page size (fixed 64KB per spec)
        wasmMemory.grow(pages);
        // .grow() takes a delta compared to the previous size
        updateMemoryViews();
        return 1;
      } catch (e) {
      }
    };

    function _emscripten_resize_heap(requestedSize) {
      requestedSize >>>= 0;
      var oldSize = HEAPU8.length;
      // With multithreaded builds, races can happen (another thread might
      // increase the size in between), so return a failure, and let the caller
      // retry. Memory resize rules:
      // 1.  Always increase heap size to at least the requested size, rounded
      // up
      //     to next page multiple.
      // 2a. If MEMORY_GROWTH_LINEAR_STEP == -1, excessively resize the heap
      //     geometrically: increase the heap size according to
      //     MEMORY_GROWTH_GEOMETRIC_STEP factor (default +20%), At most
      //     overreserve by MEMORY_GROWTH_GEOMETRIC_CAP bytes (default 96MB).
      // 2b. If MEMORY_GROWTH_LINEAR_STEP != -1, excessively resize the heap
      //     linearly: increase the heap size by at least
      //     MEMORY_GROWTH_LINEAR_STEP bytes.
      // 3.  Max size for the heap is capped at 2048MB-WASM_PAGE_SIZE, or by
      //     MAXIMUM_MEMORY, or by ASAN limit, depending on which is smallest
      // 4.  If we were unable to allocate as much memory, it may be due to
      //     over-eager decision to excessively reserve due to (3) above.
      //     Hence if an allocation fails, cut down on the amount of excess
      //     growth, in an attempt to succeed to perform a smaller allocation.
      // A limit is set for how much we can grow. We should not exceed that
      // (the wasm binary specifies it, so if we tried, we'd fail anyhow).
      var maxHeapSize = getHeapMax();
      if (requestedSize > maxHeapSize) {
        return false;
      }
      // Loop through potential heap size increases. If we attempt a too eager
      // reservation that fails, cut down on the attempted size and reserve a
      // smaller bump instead. (max 3 times, chosen somewhat arbitrarily)
      for (var cutDown = 1; cutDown <= 4; cutDown *= 2) {
        var overGrownHeapSize = oldSize * (1 + .2 / cutDown);
        // ensure geometric growth
        // but limit overreserving (default to capping at +96MB overgrowth at
        // most)
        overGrownHeapSize =
            Math.min(overGrownHeapSize, requestedSize + 100663296);
        var newSize = Math.min(
            maxHeapSize,
            alignMemory(Math.max(requestedSize, overGrownHeapSize), 65536));
        var replacement = growMemory(newSize);
        if (replacement) {
          return true;
        }
      }
      return false;
    }

    function _emscripten_stack_unwind_buffer(addr, buffer, count) {
      addr >>>= 0;
      buffer >>>= 0;
      var stack;
      if (UNWIND_CACHE.last_addr == addr) {
        stack = UNWIND_CACHE.last_stack;
      } else {
        stack = jsStackTrace().split('\n');
        if (stack[0] == 'Error') {
          stack.shift();
        }
        saveInUnwindCache(stack);
      }
      var offset = 3;
      while (stack[offset] && convertFrameToPC(stack[offset]) != addr) {
        ++offset;
      }
      for (var i = 0; i < count && stack[i + offset]; ++i) {
        HEAPU32[(((buffer) + (i * 4)) >>> 2) >>> 0] =
            convertFrameToPC(stack[i + offset]);
      }
      return i;
    }

    var GLctx;

    var webgl_enable_ANGLE_instanced_arrays = ctx => {
      // Extension available in WebGL 1 from Firefox 26 and Google Chrome 30
      // onwards. Core feature in WebGL 2.
      var ext = ctx.getExtension('ANGLE_instanced_arrays');
      // Because this extension is a core function in WebGL 2, assign the
      // extension entry points in place of where the core functions will reside
      // in WebGL 2. This way the calling code can call these without having to
      // dynamically branch depending if running against WebGL 1 or WebGL 2.
      if (ext) {
        ctx['vertexAttribDivisor'] = (index, divisor) =>
            ext['vertexAttribDivisorANGLE'](index, divisor);
        ctx['drawArraysInstanced'] = (mode, first, count, primcount) =>
            ext['drawArraysInstancedANGLE'](mode, first, count, primcount);
        ctx['drawElementsInstanced'] =
            (mode, count, type, indices, primcount) =>
                ext['drawElementsInstancedANGLE'](
                    mode, count, type, indices, primcount);
        return 1;
      }
    };

    var webgl_enable_OES_vertex_array_object = ctx => {
      // Extension available in WebGL 1 from Firefox 25 and WebKit
      // 536.28/desktop Safari 6.0.3 onwards. Core feature in WebGL 2.
      var ext = ctx.getExtension('OES_vertex_array_object');
      if (ext) {
        ctx['createVertexArray'] = () => ext['createVertexArrayOES']();
        ctx['deleteVertexArray'] = vao => ext['deleteVertexArrayOES'](vao);
        ctx['bindVertexArray'] = vao => ext['bindVertexArrayOES'](vao);
        ctx['isVertexArray'] = vao => ext['isVertexArrayOES'](vao);
        return 1;
      }
    };

    var webgl_enable_WEBGL_draw_buffers = ctx => {
      // Extension available in WebGL 1 from Firefox 28 onwards. Core feature in
      // WebGL 2.
      var ext = ctx.getExtension('WEBGL_draw_buffers');
      if (ext) {
        ctx['drawBuffers'] = (n, bufs) => ext['drawBuffersWEBGL'](n, bufs);
        return 1;
      }
    };

    var webgl_enable_WEBGL_draw_instanced_base_vertex_base_instance =
        ctx =>  // Closure is expected to be allowed to minify the '.dibvbi'
                // property, so not accessing it quoted.
        !!(ctx.dibvbi = ctx.getExtension(
               'WEBGL_draw_instanced_base_vertex_base_instance'));

    var webgl_enable_WEBGL_multi_draw_instanced_base_vertex_base_instance =
        ctx =>
            !!(ctx.mdibvbi = ctx.getExtension(
                   'WEBGL_multi_draw_instanced_base_vertex_base_instance'));

    var webgl_enable_EXT_polygon_offset_clamp = ctx =>
        !!(ctx.extPolygonOffsetClamp =
               ctx.getExtension('EXT_polygon_offset_clamp'));

    var webgl_enable_EXT_clip_control = ctx =>
        !!(ctx.extClipControl = ctx.getExtension('EXT_clip_control'));

    var webgl_enable_WEBGL_polygon_mode = ctx =>
        !!(ctx.webglPolygonMode = ctx.getExtension('WEBGL_polygon_mode'));

    var webgl_enable_WEBGL_multi_draw =
        ctx =>  // Closure is expected to be allowed to minify the
                // '.multiDrawWebgl' property, so not accessing it quoted.
        !!(ctx.multiDrawWebgl = ctx.getExtension('WEBGL_multi_draw'));

    var getEmscriptenSupportedExtensions = ctx => {
      // Restrict the list of advertised extensions to those that we actually
      // support.
      var supportedExtensions = [ // WebGL 1 extensions
  'ANGLE_instanced_arrays', 'EXT_blend_minmax', 'EXT_disjoint_timer_query', 'EXT_frag_depth', 'EXT_shader_texture_lod', 'EXT_sRGB', 'OES_element_index_uint', 'OES_fbo_render_mipmap', 'OES_standard_derivatives', 'OES_texture_float', 'OES_texture_half_float', 'OES_texture_half_float_linear', 'OES_vertex_array_object', 'WEBGL_color_buffer_float', 'WEBGL_depth_texture', 'WEBGL_draw_buffers', // WebGL 2 extensions
  'EXT_color_buffer_float', 'EXT_conservative_depth', 'EXT_disjoint_timer_query_webgl2', 'EXT_texture_norm16', 'NV_shader_noperspective_interpolation', 'WEBGL_clip_cull_distance', // WebGL 1 and WebGL 2 extensions
  'EXT_clip_control', 'EXT_color_buffer_half_float', 'EXT_depth_clamp', 'EXT_float_blend', 'EXT_polygon_offset_clamp', 'EXT_texture_compression_bptc', 'EXT_texture_compression_rgtc', 'EXT_texture_filter_anisotropic', 'KHR_parallel_shader_compile', 'OES_texture_float_linear', 'WEBGL_blend_func_extended', 'WEBGL_compressed_texture_astc', 'WEBGL_compressed_texture_etc', 'WEBGL_compressed_texture_etc1', 'WEBGL_compressed_texture_s3tc', 'WEBGL_compressed_texture_s3tc_srgb', 'WEBGL_debug_renderer_info', 'WEBGL_debug_shaders', 'WEBGL_lose_context', 'WEBGL_multi_draw', 'WEBGL_polygon_mode' ];
      // .getSupportedExtensions() can return null if context is lost, so coerce
      // to empty array.
      return ctx.getSupportedExtensions()?.filter(
                 ext => supportedExtensions.includes(ext)) ??
          [];
    };

    var registerPreMainLoop = f => {
      // Does nothing unless $MainLoop is included/used.
      typeof MainLoop != 'undefined' && MainLoop.preMainLoop.push(f);
    };

    var webglBufferSubData = (target, offset, size, data, src = HEAPU8) => {
      GLctx.bufferSubData(target, offset, src.subarray(data, data + size));
    };

    var GL = {
      counter: 1,
      buffers: [],
      mappedBuffers: {},
      programs: [],
      framebuffers: [],
      renderbuffers: [],
      textures: [],
      shaders: [],
      vaos: [],
      contexts: [],
      offscreenCanvases: {},
      queries: [],
      samplers: [],
      transformFeedbacks: [],
      syncs: [],
      byteSizeByTypeRoot: 5120,
      byteSizeByType: [1, 1, 2, 2, 4, 4, 4, 2, 3, 4, 8],
      stringCache: {},
      stringiCache: {},
      unpackAlignment: 4,
      unpackRowLength: 0,
      recordError: errorCode => {
        if (!GL.lastError) {
          GL.lastError = errorCode;
        }
      },
      getNewId: table => {
        var ret = GL.counter++;
        for (var i = table.length; i < ret; i++) {
          table[i] = null;
        }
        // Skip over any non-null elements that might have been created by
        // glBindBuffer.
        while (table[ret]) {
          ret = GL.counter++;
        }
        return ret;
      },
      genObject: (n, buffers, createFunction, objectTable) => {
        for (var i = 0; i < n; i++) {
          var buffer = GLctx[createFunction]();
          var id = buffer && GL.getNewId(objectTable);
          if (buffer) {
            buffer.name = id;
            objectTable[id] = buffer;
          } else {
            GL.recordError(1282);
          }
          HEAP32[(((buffers) + (i * 4)) >>> 2) >>> 0] = id;
        }
      },
      MAX_TEMP_BUFFER_SIZE: 2097152,
      numTempVertexBuffersPerSize: 64,
      log2ceilLookup: i => 32 - Math.clz32(i === 0 ? 0 : i - 1),
      generateTempBuffers: (quads, context) => {
        var largestIndex = GL.log2ceilLookup(GL.MAX_TEMP_BUFFER_SIZE);
        context.tempVertexBufferCounters1 = [];
        context.tempVertexBufferCounters2 = [];
        context.tempVertexBufferCounters1.length =
            context.tempVertexBufferCounters2.length = largestIndex + 1;
        context.tempVertexBuffers1 = [];
        context.tempVertexBuffers2 = [];
        context.tempVertexBuffers1.length = context.tempVertexBuffers2.length =
            largestIndex + 1;
        context.tempIndexBuffers = [];
        context.tempIndexBuffers.length = largestIndex + 1;
        for (var i = 0; i <= largestIndex; ++i) {
          context.tempIndexBuffers[i] = null;
          // Created on-demand
          context.tempVertexBufferCounters1[i] =
              context.tempVertexBufferCounters2[i] = 0;
          var ringbufferLength = GL.numTempVertexBuffersPerSize;
          context.tempVertexBuffers1[i] = [];
          context.tempVertexBuffers2[i] = [];
          var ringbuffer1 = context.tempVertexBuffers1[i];
          var ringbuffer2 = context.tempVertexBuffers2[i];
          ringbuffer1.length = ringbuffer2.length = ringbufferLength;
          for (var j = 0; j < ringbufferLength; ++j) {
            ringbuffer1[j] = ringbuffer2[j] = null;
          }
        }
        if (quads) {
          // GL_QUAD indexes can be precalculated
          context.tempQuadIndexBuffer = GLctx.createBuffer();
          context.GLctx.bindBuffer(34963, context.tempQuadIndexBuffer);
          var numIndexes = GL.MAX_TEMP_BUFFER_SIZE >> 1;
          var quadIndexes = new Uint16Array(numIndexes);
          var i = 0, v = 0;
          while (1) {
            quadIndexes[i++] = v;
            if (i >= numIndexes) break;
            quadIndexes[i++] = v + 1;
            if (i >= numIndexes) break;
            quadIndexes[i++] = v + 2;
            if (i >= numIndexes) break;
            quadIndexes[i++] = v;
            if (i >= numIndexes) break;
            quadIndexes[i++] = v + 2;
            if (i >= numIndexes) break;
            quadIndexes[i++] = v + 3;
            if (i >= numIndexes) break;
            v += 4;
          }
          context.GLctx.bufferData(34963, quadIndexes, 35044);
          context.GLctx.bindBuffer(34963, null);
        }
      },
      getTempVertexBuffer: sizeBytes => {
        var idx = GL.log2ceilLookup(sizeBytes);
        var ringbuffer = GL.currentContext.tempVertexBuffers1[idx];
        var nextFreeBufferIndex =
            GL.currentContext.tempVertexBufferCounters1[idx];
        GL.currentContext.tempVertexBufferCounters1[idx] =
            (GL.currentContext.tempVertexBufferCounters1[idx] + 1) &
            (GL.numTempVertexBuffersPerSize - 1);
        var vbo = ringbuffer[nextFreeBufferIndex];
        if (vbo) {
          return vbo;
        }
        var prevVBO = GLctx.getParameter(34964);
        ringbuffer[nextFreeBufferIndex] = GLctx.createBuffer();
        GLctx.bindBuffer(34962, ringbuffer[nextFreeBufferIndex]);
        GLctx.bufferData(34962, 1 << idx, 35048);
        GLctx.bindBuffer(34962, prevVBO);
        return ringbuffer[nextFreeBufferIndex];
      },
      getTempIndexBuffer: sizeBytes => {
        var idx = GL.log2ceilLookup(sizeBytes);
        var ibo = GL.currentContext.tempIndexBuffers[idx];
        if (ibo) {
          return ibo;
        }
        var prevIBO = GLctx.getParameter(34965);
        GL.currentContext.tempIndexBuffers[idx] = GLctx.createBuffer();
        GLctx.bindBuffer(34963, GL.currentContext.tempIndexBuffers[idx]);
        GLctx.bufferData(34963, 1 << idx, 35048);
        GLctx.bindBuffer(34963, prevIBO);
        return GL.currentContext.tempIndexBuffers[idx];
      },
      newRenderingFrameStarted: () => {
        if (!GL.currentContext) {
          return;
        }
        var vb = GL.currentContext.tempVertexBuffers1;
        GL.currentContext.tempVertexBuffers1 =
            GL.currentContext.tempVertexBuffers2;
        GL.currentContext.tempVertexBuffers2 = vb;
        vb = GL.currentContext.tempVertexBufferCounters1;
        GL.currentContext.tempVertexBufferCounters1 =
            GL.currentContext.tempVertexBufferCounters2;
        GL.currentContext.tempVertexBufferCounters2 = vb;
        var largestIndex = GL.log2ceilLookup(GL.MAX_TEMP_BUFFER_SIZE);
        for (var i = 0; i <= largestIndex; ++i) {
          GL.currentContext.tempVertexBufferCounters1[i] = 0;
        }
      },
      getSource: (shader, count, string, length) => {
        var source = '';
        for (var i = 0; i < count; ++i) {
          var len =
              length ? HEAPU32[(((length) + (i * 4)) >>> 2) >>> 0] : undefined;
          source +=
              UTF8ToString(HEAPU32[(((string) + (i * 4)) >>> 2) >>> 0], len);
        }
        return source;
      },
      calcBufLength: (size, type, stride, count) => {
        if (stride > 0) {
          return count * stride;
        }
        var typeSize = GL.byteSizeByType[type - GL.byteSizeByTypeRoot];
        return size * typeSize * count;
      },
      usedTempBuffers: [],
      preDrawHandleClientVertexAttribBindings: count => {
        GL.resetBufferBinding = false;
        // TODO: initial pass to detect ranges we need to upload, might not need
        // an upload per attrib
        for (var i = 0; i < GL.currentContext.maxVertexAttribs; ++i) {
          var cb = GL.currentContext.clientBuffers[i];
          if (!cb.clientside || !cb.enabled) continue;
          GL.resetBufferBinding = true;
          var size = GL.calcBufLength(cb.size, cb.type, cb.stride, count);
          var buf = GL.getTempVertexBuffer(size);
          GLctx.bindBuffer(34962, buf);
          webglBufferSubData(34962, 0, size, cb.ptr);
          cb.vertexAttribPointerAdaptor.call(
              GLctx, i, cb.size, cb.type, cb.normalized, cb.stride, 0);
        }
      },
      postDrawHandleClientVertexAttribBindings: () => {
        if (GL.resetBufferBinding) {
          GLctx.bindBuffer(34962, GL.buffers[GLctx.currentArrayBufferBinding]);
        }
      },
      createContext: (/** @type {HTMLCanvasElement} */ canvas,
                      webGLContextAttributes) => {
        // BUG: Workaround Safari WebGL issue: After successfully acquiring
        // WebGL context on a canvas, calling .getContext() will always return
        // that context independent of which 'webgl' or 'webgl2' context version
        // was passed. See:
        //   https://webkit.org/b/222758
        // and:
        //   https://github.com/emscripten-core/emscripten/issues/13295.
        // TODO: Once the bug is fixed and shipped in Safari, adjust the Safari
        // version field in above check.
        if (!canvas.getContextSafariWebGL2Fixed) {
          canvas.getContextSafariWebGL2Fixed = canvas.getContext;
          /**
           * @type {function(this:HTMLCanvasElement, string, (Object|null)=):
           *     (Object|null)}
           */
          function fixedGetContext(ver, attrs) {
            var gl = canvas.getContextSafariWebGL2Fixed(ver, attrs);
            return ((ver == 'webgl') == (gl instanceof WebGLRenderingContext)) ?
                gl :
                null;
          }
          canvas.getContext = fixedGetContext;
        }
        var ctx = (webGLContextAttributes.majorVersion > 1) ?
            canvas.getContext('webgl2', webGLContextAttributes) :
            canvas.getContext('webgl', webGLContextAttributes);
        if (!ctx) return 0;
        var handle = GL.registerContext(ctx, webGLContextAttributes);
        return handle;
      },
      registerContext: (ctx, webGLContextAttributes) => {
        // without pthreads a context is just an integer ID
        var handle = GL.getNewId(GL.contexts);
        var context = {
          handle,
          attributes: webGLContextAttributes,
          version: webGLContextAttributes.majorVersion,
          GLctx: ctx
        };
        // Store the created context object so that we can access the context
        // given a canvas without having to pass the parameters again.
        if (ctx.canvas) ctx.canvas.GLctxObject = context;
        GL.contexts[handle] = context;
        if (typeof webGLContextAttributes.enableExtensionsByDefault ==
                'undefined' ||
            webGLContextAttributes.enableExtensionsByDefault) {
          GL.initExtensions(context);
        }
        context.maxVertexAttribs = context.GLctx.getParameter(34921);
        context.clientBuffers = [];
        for (var i = 0; i < context.maxVertexAttribs; i++) {
          context.clientBuffers[i] = {
            enabled: false,
            clientside: false,
            size: 0,
            type: 0,
            normalized: 0,
            stride: 0,
            ptr: 0,
            vertexAttribPointerAdaptor: null
          };
        }
        GL.generateTempBuffers(false, context);
        return handle;
      },
      makeContextCurrent: contextHandle => {
        // Active Emscripten GL layer context object.
        GL.currentContext = GL.contexts[contextHandle];
        // Active WebGL context object.
        Module['ctx'] = GLctx = GL.currentContext?.GLctx;
        return !(contextHandle && !GLctx);
      },
      getContext: contextHandle => GL.contexts[contextHandle],
      deleteContext: contextHandle => {
        if (GL.currentContext === GL.contexts[contextHandle]) {
          GL.currentContext = null;
        }
        if (typeof JSEvents == 'object') {
          // Release all JS event handlers on the DOM element that the GL
          // context is associated with since the context is now deleted.
          JSEvents.removeAllHandlersOnTarget(
              GL.contexts[contextHandle].GLctx.canvas);
        }
        // Make sure the canvas object no longer refers to the context object so
        // there are no GC surprises.
        if (GL.contexts[contextHandle]?.GLctx.canvas) {
          GL.contexts[contextHandle].GLctx.canvas.GLctxObject = undefined;
        }
        GL.contexts[contextHandle] = null;
      },
      initExtensions: context => {
        // If this function is called without a specific context object, init
        // the extensions of the currently active context.
        context ||= GL.currentContext;
        if (context.initExtensionsDone) return;
        context.initExtensionsDone = true;
        var GLctx = context.GLctx;
        // Detect the presence of a few extensions manually, since the GL
        // interop layer itself will need to know if they exist. Extensions that
        // are available in both WebGL 1 and WebGL 2
        webgl_enable_WEBGL_multi_draw(GLctx);
        webgl_enable_EXT_polygon_offset_clamp(GLctx);
        webgl_enable_EXT_clip_control(GLctx);
        webgl_enable_WEBGL_polygon_mode(GLctx);
        // Extensions that are only available in WebGL 1 (the calls will be
        // no-ops if called on a WebGL 2 context active)
        webgl_enable_ANGLE_instanced_arrays(GLctx);
        webgl_enable_OES_vertex_array_object(GLctx);
        webgl_enable_WEBGL_draw_buffers(GLctx);
        // Extensions that are available from WebGL >= 2 (no-op if called on a
        // WebGL 1 context active)
        webgl_enable_WEBGL_draw_instanced_base_vertex_base_instance(GLctx);
        webgl_enable_WEBGL_multi_draw_instanced_base_vertex_base_instance(
            GLctx);
        // On WebGL 2, EXT_disjoint_timer_query is replaced with an alternative
        // that's based on core APIs, and exposes only the queryCounterEXT()
        // entrypoint.
        if (context.version >= 2) {
          GLctx.disjointTimerQueryExt =
              GLctx.getExtension('EXT_disjoint_timer_query_webgl2');
        }
        // However, Firefox exposes the WebGL 1 version on WebGL 2 as well and
        // thus we look for the WebGL 1 version again if the WebGL 2 version
        // isn't present. https://bugzil.la/1328882
        if (context.version < 2 || !GLctx.disjointTimerQueryExt) {
          GLctx.disjointTimerQueryExt =
              GLctx.getExtension('EXT_disjoint_timer_query');
        }
        for (var ext of getEmscriptenSupportedExtensions(GLctx)) {
          // WEBGL_lose_context, WEBGL_debug_renderer_info and
          // WEBGL_debug_shaders are not enabled by default.
          if (!ext.includes('lose_context') && !ext.includes('debug')) {
            // Call .getExtension() to enable that extension permanently.
            GLctx.getExtension(ext);
          }
        }
      }
    };

    var webglPowerPreferences = ['default', 'low-power', 'high-performance'];

    /** @type {Object} */ var specialHTMLTargets =
        [0, globalThis.document ?? 0, globalThis.window ?? 0];

    var findEventTarget = target => {
      // The sensible 'default' target varies between events, but use window as
      // the default since DOM events mostly can default to that. Specific
      // callback registrations override their own defaults.
      if (!target) return window;
      if (typeof target == 'number')
        target = specialHTMLTargets[target] || UTF8ToString(target);
      if (target === '#window')
        return window;
      else if (target === '#document')
        return document;
      else if (target === '#screen')
        return screen;
      else if (target === '#canvas')
        return Module['canvas'];
      else if (typeof target == 'string')
        return globalThis.document?.getElementById(target);
      return target;
    };

    var findCanvasEventTarget = target => {
      if (typeof target == 'number') target = UTF8ToString(target);
      if (!target || target === '#canvas') {
        if (typeof GL != 'undefined' && GL.offscreenCanvases['canvas'])
          return GL.offscreenCanvases['canvas'];
        // TODO: Remove this line, target '#canvas' should refer only to
        // Module['canvas'], not to GL.offscreenCanvases['canvas'] - but need
        // stricter tests to be able to remove this line.
        return Module['canvas'];
      }
      if (typeof GL != 'undefined' && GL.offscreenCanvases[target])
        return GL.offscreenCanvases[target];
      return findEventTarget(target);
    };

    function _emscripten_webgl_do_create_context(target, attributes) {
      target >>>= 0;
      attributes >>>= 0;
      var attr32 = ((attributes) >>> 2);
      var powerPreference = HEAP32[attr32 + (8 >> 2) >>> 0];
      var contextAttributes = {
        'alpha': !!HEAP8[attributes + 0 >>> 0],
        'depth': !!HEAP8[attributes + 1 >>> 0],
        'stencil': !!HEAP8[attributes + 2 >>> 0],
        'antialias': !!HEAP8[attributes + 3 >>> 0],
        'premultipliedAlpha': !!HEAP8[attributes + 4 >>> 0],
        'preserveDrawingBuffer': !!HEAP8[attributes + 5 >>> 0],
        'powerPreference': webglPowerPreferences[powerPreference],
        'failIfMajorPerformanceCaveat': !!HEAP8[attributes + 12 >>> 0],
        'desynchronized': !!HEAP8[attributes + 33 >>> 0],
        // The following are not predefined WebGL context attributes in the
        // WebGL specification, so the property names can be minified by
        // Closure.
        majorVersion: HEAP32[attr32 + (16 >> 2) >>> 0],
        minorVersion: HEAP32[attr32 + (20 >> 2) >>> 0],
        enableExtensionsByDefault: HEAP8[attributes + 24 >>> 0],
        explicitSwapControl: HEAP8[attributes + 25 >>> 0],
        proxyContextToMainThread: HEAP32[attr32 + (28 >> 2) >>> 0],
        renderViaOffscreenBackBuffer: HEAP8[attributes + 32 >>> 0]
      };
      var canvas = findCanvasEventTarget(target);
      if (!canvas) {
        return 0;
      }
      if (contextAttributes.explicitSwapControl) {
        return 0;
      }
      var contextHandle = GL.createContext(canvas, contextAttributes);
      return contextHandle;
    }

    var _emscripten_webgl_create_context = _emscripten_webgl_do_create_context;

    function _emscripten_webgl_destroy_context(contextHandle) {
      contextHandle >>>= 0;
      if (GL.currentContext == contextHandle) GL.currentContext = 0;
      GL.deleteContext(contextHandle);
    }

    function _emscripten_webgl_get_context_attributes(c, a) {
      c >>>= 0;
      a >>>= 0;
      if (!a) return -5;
      c = GL.contexts[c];
      if (!c) return -3;
      var t = c.GLctx?.getContextAttributes();
      if (!t) return -3;
      HEAP8[a >>> 0] = t.alpha;
      HEAP8[(a) + (1) >>> 0] = t.depth;
      HEAP8[(a) + (2) >>> 0] = t.stencil;
      HEAP8[(a) + (3) >>> 0] = t.antialias;
      HEAP8[(a) + (4) >>> 0] = t.premultipliedAlpha;
      HEAP8[(a) + (5) >>> 0] = t.preserveDrawingBuffer;
      var power = t['powerPreference'] &&
          webglPowerPreferences.indexOf(t['powerPreference']);
      HEAP32[(((a) + (8)) >>> 2) >>> 0] = power;
      HEAP8[(a) + (12) >>> 0] = t.failIfMajorPerformanceCaveat;
      HEAP8[(a) + (33) >>> 0] = t.desynchronized;
      HEAP32[(((a) + (16)) >>> 2) >>> 0] = c.version;
      HEAP32[(((a) + (20)) >>> 2) >>> 0] = 0;
      HEAP8[(a) + (24) >>> 0] = c.attributes.enableExtensionsByDefault;
      return 0;
    }

    function _emscripten_webgl_do_get_current_context() {
      return GL.currentContext ? GL.currentContext.handle : 0;
    }

    var _emscripten_webgl_get_current_context =
        _emscripten_webgl_do_get_current_context;

    function _emscripten_webgl_make_context_current(contextHandle) {
      contextHandle >>>= 0;
      var success = GL.makeContextCurrent(contextHandle);
      return success ? 0 : -5;
    }

    var stackAlloc = sz => __emscripten_stack_alloc(sz);

    var stringToUTF8OnStack = str => {
      var size = lengthBytesUTF8(str) + 1;
      var ret = stackAlloc(size);
      stringToUTF8(str, ret, size);
      return ret;
    };

    var writeI53ToI64 = (ptr, num) => {
      HEAPU32[((ptr) >>> 2) >>> 0] = num;
      var lower = HEAPU32[((ptr) >>> 2) >>> 0];
      HEAPU32[(((ptr) + (4)) >>> 2) >>> 0] = (num - lower) / 4294967296;
    };

    var WebGPU = {
      Internals: {
        jsObjects: [],
        jsObjectInsert: (ptr, jsObject) => {
          ptr >>>= 0;
          WebGPU.Internals.jsObjects[ptr] = jsObject;
        },
        bufferOnUnmaps: [],
        futures: [],
        futureInsert: (futureId, promise) => {
          WebGPU.Internals.futures[futureId] =
              new Promise(resolve => promise.finally(() => resolve(futureId)));
        }
      },
      getJsObject: ptr => {
        if (!ptr) return undefined;
        ptr >>>= 0;
        return WebGPU.Internals.jsObjects[ptr];
      },
      importJsAdapter: (obj, parentPtr = 0) => {
        var ptr = _emwgpuCreateAdapter(parentPtr);
        WebGPU.Internals.jsObjects[ptr] = obj;
        return ptr;
      },
      importJsBindGroup: (obj, parentPtr = 0) => {
        var ptr = _emwgpuCreateBindGroup(parentPtr);
        WebGPU.Internals.jsObjects[ptr] = obj;
        return ptr;
      },
      importJsBindGroupLayout: (obj, parentPtr = 0) => {
        var ptr = _emwgpuCreateBindGroupLayout(parentPtr);
        WebGPU.Internals.jsObjects[ptr] = obj;
        return ptr;
      },
      importJsBuffer: (buffer, parentPtr = 0) => {
        // At the moment, we do not allow importing pending buffers.
        assert(buffer.mapState === 'unmapped');
        var bufferPtr = _emwgpuImportBuffer(parentPtr);
        WebGPU.Internals.jsObjectInsert(bufferPtr, buffer);
        return bufferPtr;
      },
      importJsCommandBuffer: (obj, parentPtr = 0) => {
        var ptr = _emwgpuCreateCommandBuffer(parentPtr);
        WebGPU.Internals.jsObjects[ptr] = obj;
        return ptr;
      },
      importJsCommandEncoder: (obj, parentPtr = 0) => {
        var ptr = _emwgpuCreateCommandEncoder(parentPtr);
        WebGPU.Internals.jsObjects[ptr] = obj;
        return ptr;
      },
      importJsComputePassEncoder: (obj, parentPtr = 0) => {
        var ptr = _emwgpuCreateComputePassEncoder(parentPtr);
        WebGPU.Internals.jsObjects[ptr] = obj;
        return ptr;
      },
      importJsComputePipeline: (obj, parentPtr = 0) => {
        var ptr = _emwgpuCreateComputePipeline(parentPtr);
        WebGPU.Internals.jsObjects[ptr] = obj;
        return ptr;
      },
      importJsDevice: (device, parentPtr = 0) => {
        var queuePtr = _emwgpuCreateQueue(parentPtr);
        var devicePtr = _emwgpuCreateDevice(parentPtr, queuePtr);
        WebGPU.Internals.jsObjectInsert(queuePtr, device.queue);
        WebGPU.Internals.jsObjectInsert(devicePtr, device);
        return devicePtr;
      },
      importJsExternalTexture: (obj, parentPtr = 0) => {
        var ptr = _emwgpuCreateExternalTexture(parentPtr);
        WebGPU.Internals.jsObjects[ptr] = obj;
        return ptr;
      },
      importJsPipelineLayout: (obj, parentPtr = 0) => {
        var ptr = _emwgpuCreatePipelineLayout(parentPtr);
        WebGPU.Internals.jsObjects[ptr] = obj;
        return ptr;
      },
      importJsQuerySet: (obj, parentPtr = 0) => {
        var ptr = _emwgpuCreateQuerySet(parentPtr);
        WebGPU.Internals.jsObjects[ptr] = obj;
        return ptr;
      },
      importJsQueue: (obj, parentPtr = 0) => {
        var ptr = _emwgpuCreateQueue(parentPtr);
        WebGPU.Internals.jsObjects[ptr] = obj;
        return ptr;
      },
      importJsRenderBundle: (obj, parentPtr = 0) => {
        var ptr = _emwgpuCreateRenderBundle(parentPtr);
        WebGPU.Internals.jsObjects[ptr] = obj;
        return ptr;
      },
      importJsRenderBundleEncoder: (obj, parentPtr = 0) => {
        var ptr = _emwgpuCreateRenderBundleEncoder(parentPtr);
        WebGPU.Internals.jsObjects[ptr] = obj;
        return ptr;
      },
      importJsRenderPassEncoder: (obj, parentPtr = 0) => {
        var ptr = _emwgpuCreateRenderPassEncoder(parentPtr);
        WebGPU.Internals.jsObjects[ptr] = obj;
        return ptr;
      },
      importJsRenderPipeline: (obj, parentPtr = 0) => {
        var ptr = _emwgpuCreateRenderPipeline(parentPtr);
        WebGPU.Internals.jsObjects[ptr] = obj;
        return ptr;
      },
      importJsSampler: (obj, parentPtr = 0) => {
        var ptr = _emwgpuCreateSampler(parentPtr);
        WebGPU.Internals.jsObjects[ptr] = obj;
        return ptr;
      },
      importJsShaderModule: (obj, parentPtr = 0) => {
        var ptr = _emwgpuCreateShaderModule(parentPtr);
        WebGPU.Internals.jsObjects[ptr] = obj;
        return ptr;
      },
      importJsSurface: (obj, parentPtr = 0) => {
        var ptr = _emwgpuCreateSurface(parentPtr);
        WebGPU.Internals.jsObjects[ptr] = obj;
        return ptr;
      },
      importJsTexture: (obj, parentPtr = 0) => {
        var ptr = _emwgpuCreateTexture(parentPtr);
        WebGPU.Internals.jsObjects[ptr] = obj;
        return ptr;
      },
      importJsTextureView: (obj, parentPtr = 0) => {
        var ptr = _emwgpuCreateTextureView(parentPtr);
        WebGPU.Internals.jsObjects[ptr] = obj;
        return ptr;
      },
      errorCallback: (callback, type, message, userdata) => {
        var sp = stackSave();
        var messagePtr = stringToUTF8OnStack(message);
        ((a1, a2, a3) => dynCall_viii(callback, a1, a2, a3))(
            type, messagePtr, userdata);
        stackRestore(sp);
      },
      iterateExtensions: (root, handlers) => {
        for (var ptr = HEAPU32[((root) >>> 2) >>> 0]; ptr;
             ptr = HEAPU32[((ptr) >>> 2) >>> 0]) {
          var sType = HEAP32[(((ptr) + (4)) >>> 2) >>> 0];
          // This will crash if there's no handler indicating either a bogus
          // sType, or one we haven't implemented yet.
          var handler = handlers[sType](ptr);
        }
      },
      setStringView: (ptr, data, length) => {
        HEAPU32[((ptr) >>> 2) >>> 0] = data;
        HEAPU32[(((ptr) + (4)) >>> 2) >>> 0] = length;
      },
      makeStringFromStringView: stringViewPtr => {
        var ptr = HEAPU32[((stringViewPtr) >>> 2) >>> 0];
        var length = HEAPU32[(((stringViewPtr) + (4)) >>> 2) >>> 0];
        // UTF8ToString stops at the first null terminator character in the
        // string regardless of the length.
        return UTF8ToString(ptr, length);
      },
      makeStringFromOptionalStringView: stringViewPtr => {
        var ptr = HEAPU32[((stringViewPtr) >>> 2) >>> 0];
        var length = HEAPU32[(((stringViewPtr) + (4)) >>> 2) >>> 0];
        // If we don't have a valid string pointer, just return undefined when
        // optional.
        if (!ptr) {
          if (length === 0) {
            return '';
          }
          return undefined;
        }
        // UTF8ToString stops at the first null terminator character in the
        // string regardless of the length.
        return UTF8ToString(ptr, length);
      },
      makeColor: ptr => ({
        'r': HEAPF64[((ptr) >>> 3) >>> 0],
        'g': HEAPF64[(((ptr) + (8)) >>> 3) >>> 0],
        'b': HEAPF64[(((ptr) + (16)) >>> 3) >>> 0],
        'a': HEAPF64[(((ptr) + (24)) >>> 3) >>> 0]
      }),
      makeExtent3D: ptr => ({
        'width': HEAPU32[((ptr) >>> 2) >>> 0],
        'height': HEAPU32[(((ptr) + (4)) >>> 2) >>> 0],
        'depthOrArrayLayers': HEAPU32[(((ptr) + (8)) >>> 2) >>> 0]
      }),
      makeOrigin3D: ptr => ({
        'x': HEAPU32[((ptr) >>> 2) >>> 0],
        'y': HEAPU32[(((ptr) + (4)) >>> 2) >>> 0],
        'z': HEAPU32[(((ptr) + (8)) >>> 2) >>> 0]
      }),
      makeTexelCopyTextureInfo: ptr => ({
        'texture': WebGPU.getJsObject(HEAPU32[((ptr) >>> 2) >>> 0]),
        'mipLevel': HEAPU32[(((ptr) + (4)) >>> 2) >>> 0],
        'origin': WebGPU.makeOrigin3D(ptr + 8),
        'aspect': WebGPU.TextureAspect[HEAP32[(((ptr) + (20)) >>> 2) >>> 0]]
      }),
      makeTexelCopyBufferLayout: ptr => {
        var bytesPerRow = HEAPU32[(((ptr) + (8)) >>> 2) >>> 0];
        var rowsPerImage = HEAPU32[(((ptr) + (12)) >>> 2) >>> 0];
        return {
          'offset': readI53FromI64(ptr),
          'bytesPerRow': bytesPerRow === 4294967295 ? undefined : bytesPerRow,
          'rowsPerImage': rowsPerImage === 4294967295 ? undefined : rowsPerImage
        };
      },
      makeTexelCopyBufferInfo: ptr => {
        var layoutPtr = ptr + 0;
        var bufferCopyView = WebGPU.makeTexelCopyBufferLayout(layoutPtr);
        bufferCopyView['buffer'] =
            WebGPU.getJsObject(HEAPU32[(((ptr) + (16)) >>> 2) >>> 0]);
        return bufferCopyView;
      },
      makePassTimestampWrites: ptr => {
        if (ptr === 0) return undefined;
        return {
          'querySet': WebGPU.getJsObject(HEAPU32[(((ptr) + (4)) >>> 2) >>> 0]),
          'beginningOfPassWriteIndex': HEAPU32[(((ptr) + (8)) >>> 2) >>> 0],
          'endOfPassWriteIndex': HEAPU32[(((ptr) + (12)) >>> 2) >>> 0]
        };
      },
      makePipelineConstants: (constantCount, constantsPtr) => {
        if (!constantCount) return;
        var constants = {};
        for (var i = 0; i < constantCount; ++i) {
          var entryPtr = constantsPtr + 24 * i;
          var key = WebGPU.makeStringFromStringView(entryPtr + 4);
          constants[key] = HEAPF64[(((entryPtr) + (16)) >>> 3) >>> 0];
        }
        return constants;
      },
      makePipelineLayout: layoutPtr => {
        if (!layoutPtr) return 'auto';
        return WebGPU.getJsObject(layoutPtr);
      },
      makeComputeState: ptr => {
        if (!ptr) return undefined;
        var desc = {
          'module': WebGPU.getJsObject(HEAPU32[(((ptr) + (4)) >>> 2) >>> 0]),
          'constants': WebGPU.makePipelineConstants(
              HEAPU32[(((ptr) + (16)) >>> 2) >>> 0],
              HEAPU32[(((ptr) + (20)) >>> 2) >>> 0]),
          'entryPoint': WebGPU.makeStringFromOptionalStringView(ptr + 8)
        };
        return desc;
      },
      makeComputePipelineDesc: descriptor => {
        var desc = {
          'label': WebGPU.makeStringFromOptionalStringView(descriptor + 4),
          'layout': WebGPU.makePipelineLayout(
              HEAPU32[(((descriptor) + (12)) >>> 2) >>> 0]),
          'compute': WebGPU.makeComputeState(descriptor + 16)
        };
        return desc;
      },
      makeRenderPipelineDesc: descriptor => {
        function makePrimitiveState(psPtr) {
          if (!psPtr) return undefined;
          return {
            'topology':
                WebGPU.PrimitiveTopology[HEAP32[(((psPtr) + (4)) >>> 2) >>> 0]],
            'stripIndexFormat':
                WebGPU.IndexFormat[HEAP32[(((psPtr) + (8)) >>> 2) >>> 0]],
            'frontFace':
                WebGPU.FrontFace[HEAP32[(((psPtr) + (12)) >>> 2) >>> 0]],
            'cullMode': WebGPU.CullMode[HEAP32[(((psPtr) + (16)) >>> 2) >>> 0]],
            'unclippedDepth': !!(HEAPU32[(((psPtr) + (20)) >>> 2) >>> 0])
          };
        }
        function makeBlendComponent(bdPtr) {
          if (!bdPtr) return undefined;
          return {
            'operation': WebGPU.BlendOperation[HEAP32[((bdPtr) >>> 2) >>> 0]],
            'srcFactor':
                WebGPU.BlendFactor[HEAP32[(((bdPtr) + (4)) >>> 2) >>> 0]],
            'dstFactor':
                WebGPU.BlendFactor[HEAP32[(((bdPtr) + (8)) >>> 2) >>> 0]]
          };
        }
        function makeBlendState(bsPtr) {
          if (!bsPtr) return undefined;
          return {
            'alpha': makeBlendComponent(bsPtr + 12),
            'color': makeBlendComponent(bsPtr + 0)
          };
        }
        function makeColorState(csPtr) {
          var format =
              WebGPU.TextureFormat[HEAP32[(((csPtr) + (4)) >>> 2) >>> 0]];
          return format ? {
            'format': format,
            'blend': makeBlendState(HEAPU32[(((csPtr) + (8)) >>> 2) >>> 0]),
            'writeMask': HEAPU32[(((csPtr) + (16)) >>> 2) >>> 0]
          } :
                          undefined;
        }
        function makeColorStates(count, csArrayPtr) {
          var states = [];
          for (var i = 0; i < count; ++i) {
            states.push(makeColorState(csArrayPtr + 24 * i));
          }
          return states;
        }
        function makeStencilStateFace(ssfPtr) {
          return {
            'compare': WebGPU.CompareFunction[HEAP32[((ssfPtr) >>> 2) >>> 0]],
            'failOp':
                WebGPU.StencilOperation[HEAP32[(((ssfPtr) + (4)) >>> 2) >>> 0]],
            'depthFailOp':
                WebGPU.StencilOperation[HEAP32[(((ssfPtr) + (8)) >>> 2) >>> 0]],
            'passOp':
                WebGPU.StencilOperation[HEAP32[(((ssfPtr) + (12)) >>> 2) >>> 0]]
          };
        }
        function makeDepthStencilState(dssPtr) {
          if (!dssPtr) return undefined;
          return {
            'format':
                WebGPU.TextureFormat[HEAP32[(((dssPtr) + (4)) >>> 2) >>> 0]],
            'depthWriteEnabled': !!(HEAPU32[(((dssPtr) + (8)) >>> 2) >>> 0]),
            'depthCompare':
                WebGPU.CompareFunction[HEAP32[(((dssPtr) + (12)) >>> 2) >>> 0]],
            'stencilFront': makeStencilStateFace(dssPtr + 16),
            'stencilBack': makeStencilStateFace(dssPtr + 32),
            'stencilReadMask': HEAPU32[(((dssPtr) + (48)) >>> 2) >>> 0],
            'stencilWriteMask': HEAPU32[(((dssPtr) + (52)) >>> 2) >>> 0],
            'depthBias': HEAP32[(((dssPtr) + (56)) >>> 2) >>> 0],
            'depthBiasSlopeScale': HEAPF32[(((dssPtr) + (60)) >>> 2) >>> 0],
            'depthBiasClamp': HEAPF32[(((dssPtr) + (64)) >>> 2) >>> 0]
          };
        }
        function makeVertexAttribute(vaPtr) {
          return {
            'format':
                WebGPU.VertexFormat[HEAP32[(((vaPtr) + (4)) >>> 2) >>> 0]],
            'offset': readI53FromI64((vaPtr) + (8)),
            'shaderLocation': HEAPU32[(((vaPtr) + (16)) >>> 2) >>> 0]
          };
        }
        function makeVertexAttributes(count, vaArrayPtr) {
          var vas = [];
          for (var i = 0; i < count; ++i) {
            vas.push(makeVertexAttribute(vaArrayPtr + i * 24));
          }
          return vas;
        }
        function makeVertexBuffer(vbPtr) {
          if (!vbPtr) return undefined;
          var stepMode =
              WebGPU.VertexStepMode[HEAP32[(((vbPtr) + (4)) >>> 2) >>> 0]];
          var attributeCount = HEAPU32[(((vbPtr) + (16)) >>> 2) >>> 0];
          if (!stepMode && !attributeCount) {
            return null;
          }
          return {
            'arrayStride': readI53FromI64((vbPtr) + (8)),
            'stepMode': stepMode,
            'attributes': makeVertexAttributes(
                attributeCount, HEAPU32[(((vbPtr) + (20)) >>> 2) >>> 0])
          };
        }
        function makeVertexBuffers(count, vbArrayPtr) {
          if (!count) return undefined;
          var vbs = [];
          for (var i = 0; i < count; ++i) {
            vbs.push(makeVertexBuffer(vbArrayPtr + i * 24));
          }
          return vbs;
        }
        function makeVertexState(viPtr) {
          if (!viPtr) return undefined;
          var desc = {
            'module':
                WebGPU.getJsObject(HEAPU32[(((viPtr) + (4)) >>> 2) >>> 0]),
            'constants': WebGPU.makePipelineConstants(
                HEAPU32[(((viPtr) + (16)) >>> 2) >>> 0],
                HEAPU32[(((viPtr) + (20)) >>> 2) >>> 0]),
            'buffers': makeVertexBuffers(
                HEAPU32[(((viPtr) + (24)) >>> 2) >>> 0],
                HEAPU32[(((viPtr) + (28)) >>> 2) >>> 0]),
            'entryPoint': WebGPU.makeStringFromOptionalStringView(viPtr + 8)
          };
          return desc;
        }
        function makeMultisampleState(msPtr) {
          if (!msPtr) return undefined;
          return {
            'count': HEAPU32[(((msPtr) + (4)) >>> 2) >>> 0],
            'mask': HEAPU32[(((msPtr) + (8)) >>> 2) >>> 0],
            'alphaToCoverageEnabled':
                !!(HEAPU32[(((msPtr) + (12)) >>> 2) >>> 0])
          };
        }
        function makeFragmentState(fsPtr) {
          if (!fsPtr) return undefined;
          var desc = {
            'module':
                WebGPU.getJsObject(HEAPU32[(((fsPtr) + (4)) >>> 2) >>> 0]),
            'constants': WebGPU.makePipelineConstants(
                HEAPU32[(((fsPtr) + (16)) >>> 2) >>> 0],
                HEAPU32[(((fsPtr) + (20)) >>> 2) >>> 0]),
            'targets': makeColorStates(
                HEAPU32[(((fsPtr) + (24)) >>> 2) >>> 0],
                HEAPU32[(((fsPtr) + (28)) >>> 2) >>> 0]),
            'entryPoint': WebGPU.makeStringFromOptionalStringView(fsPtr + 8)
          };
          return desc;
        }
        var desc = {
          'label': WebGPU.makeStringFromOptionalStringView(descriptor + 4),
          'layout': WebGPU.makePipelineLayout(
              HEAPU32[(((descriptor) + (12)) >>> 2) >>> 0]),
          'vertex': makeVertexState(descriptor + 16),
          'primitive': makePrimitiveState(descriptor + 48),
          'depthStencil': makeDepthStencilState(
              HEAPU32[(((descriptor) + (72)) >>> 2) >>> 0]),
          'multisample': makeMultisampleState(descriptor + 76),
          'fragment':
              makeFragmentState(HEAPU32[(((descriptor) + (92)) >>> 2) >>> 0])
        };
        return desc;
      },
      fillLimitStruct: (limits, limitsOutPtr) => {
        var nextInChainPtr = HEAPU32[((limitsOutPtr) >>> 2) >>> 0];
        function setLimitValueU32(
            name, basePtr, limitOffset, fallbackValue = 0) {
          var limitValue = limits[name] ?? fallbackValue;
          HEAPU32[(((basePtr) + (limitOffset)) >>> 2) >>> 0] = limitValue;
        }
        function setLimitValueU64(
            name, basePtr, limitOffset, fallbackValue = 0) {
          var limitValue = limits[name] ?? fallbackValue;
          // Limits are integer-valued JS `Number`s, so they fit in 'i53'.
          writeI53ToI64((basePtr) + (limitOffset), limitValue);
        }
        setLimitValueU32('maxTextureDimension1D', limitsOutPtr, 4);
        setLimitValueU32('maxTextureDimension2D', limitsOutPtr, 8);
        setLimitValueU32('maxTextureDimension3D', limitsOutPtr, 12);
        setLimitValueU32('maxTextureArrayLayers', limitsOutPtr, 16);
        setLimitValueU32('maxBindGroups', limitsOutPtr, 20);
        setLimitValueU32('maxBindGroupsPlusVertexBuffers', limitsOutPtr, 24);
        setLimitValueU32('maxBindingsPerBindGroup', limitsOutPtr, 28);
        setLimitValueU32(
            'maxDynamicUniformBuffersPerPipelineLayout', limitsOutPtr, 32);
        setLimitValueU32(
            'maxDynamicStorageBuffersPerPipelineLayout', limitsOutPtr, 36);
        setLimitValueU32('maxSampledTexturesPerShaderStage', limitsOutPtr, 40);
        setLimitValueU32('maxSamplersPerShaderStage', limitsOutPtr, 44);
        setLimitValueU32('maxStorageBuffersPerShaderStage', limitsOutPtr, 48);
        setLimitValueU32('maxStorageTexturesPerShaderStage', limitsOutPtr, 52);
        setLimitValueU32('maxUniformBuffersPerShaderStage', limitsOutPtr, 56);
        setLimitValueU32('minUniformBufferOffsetAlignment', limitsOutPtr, 80);
        setLimitValueU32('minStorageBufferOffsetAlignment', limitsOutPtr, 84);
        setLimitValueU64('maxUniformBufferBindingSize', limitsOutPtr, 64);
        setLimitValueU64('maxStorageBufferBindingSize', limitsOutPtr, 72);
        setLimitValueU32('maxVertexBuffers', limitsOutPtr, 88);
        setLimitValueU64('maxBufferSize', limitsOutPtr, 96);
        setLimitValueU32('maxVertexAttributes', limitsOutPtr, 104);
        setLimitValueU32('maxVertexBufferArrayStride', limitsOutPtr, 108);
        setLimitValueU32('maxInterStageShaderVariables', limitsOutPtr, 112);
        setLimitValueU32('maxColorAttachments', limitsOutPtr, 116);
        setLimitValueU32('maxColorAttachmentBytesPerSample', limitsOutPtr, 120);
        setLimitValueU32('maxComputeWorkgroupStorageSize', limitsOutPtr, 124);
        setLimitValueU32(
            'maxComputeInvocationsPerWorkgroup', limitsOutPtr, 128);
        setLimitValueU32('maxComputeWorkgroupSizeX', limitsOutPtr, 132);
        setLimitValueU32('maxComputeWorkgroupSizeY', limitsOutPtr, 136);
        setLimitValueU32('maxComputeWorkgroupSizeZ', limitsOutPtr, 140);
        setLimitValueU32('maxComputeWorkgroupsPerDimension', limitsOutPtr, 144);
        // Note this limit is new and won't be present in all browsers for a
        // while. Fall back to 0.
        setLimitValueU32('maxImmediateSize', limitsOutPtr, 148);
        if (nextInChainPtr !== 0) {
          var sType = HEAP32[(((nextInChainPtr) + (4)) >>> 2) >>> 0];
          var compatibilityModeLimitsPtr = nextInChainPtr;
          // Note these limits are new and won't be present in all browsers for
          // a while. Fall back to exposing the PerShaderStage limit.
          setLimitValueU32(
              'maxStorageBuffersInVertexStage', compatibilityModeLimitsPtr, 8,
              limits.maxStorageBuffersPerShaderStage);
          setLimitValueU32(
              'maxStorageBuffersInFragmentStage', compatibilityModeLimitsPtr,
              16, limits.maxStorageBuffersPerShaderStage);
          setLimitValueU32(
              'maxStorageTexturesInVertexStage', compatibilityModeLimitsPtr, 12,
              limits.maxStorageTexturesPerShaderStage);
          setLimitValueU32(
              'maxStorageTexturesInFragmentStage', compatibilityModeLimitsPtr,
              20, limits.maxStorageTexturesPerShaderStage);
        }
      },
      fillAdapterInfoStruct: (info, infoStruct) => {
        // Populate subgroup limits.
        HEAPU32[(((infoStruct) + (52)) >>> 2) >>> 0] = info.subgroupMinSize;
        HEAPU32[(((infoStruct) + (56)) >>> 2) >>> 0] = info.subgroupMaxSize;
        // Append all the strings together to condense into a single malloc.
        var strs =
            info.vendor + info.architecture + info.device + info.description;
        var strPtr = stringToNewUTF8(strs);
        var vendorLen = lengthBytesUTF8(info.vendor);
        WebGPU.setStringView(infoStruct + 4, strPtr, vendorLen);
        strPtr += vendorLen;
        var architectureLen = lengthBytesUTF8(info.architecture);
        WebGPU.setStringView(infoStruct + 12, strPtr, architectureLen);
        strPtr += architectureLen;
        var deviceLen = lengthBytesUTF8(info.device);
        WebGPU.setStringView(infoStruct + 20, strPtr, deviceLen);
        strPtr += deviceLen;
        var descriptionLen = lengthBytesUTF8(info.description);
        WebGPU.setStringView(infoStruct + 28, strPtr, descriptionLen);
        strPtr += descriptionLen;
        HEAP32[(((infoStruct) + (36)) >>> 2) >>> 0] = 2;
        var adapterType = info.isFallbackAdapter ? 3 : 4;
        HEAP32[(((infoStruct) + (40)) >>> 2) >>> 0] = adapterType;
        HEAPU32[(((infoStruct) + (44)) >>> 2) >>> 0] = 0;
        HEAPU32[(((infoStruct) + (48)) >>> 2) >>> 0] = 0;
        WebGPU.iterateExtensions(infoStruct, {
          327739: ext => {
            let configCount = 0;
            let configsPtr = 0;
            if (info.subgroupMatrixConfigs) {
              configCount = info.subgroupMatrixConfigs.length;
              configsPtr = _malloc(configCount * 20);
              for (const [i, cfg] of info.subgroupMatrixConfigs.entries()) {
                const ptr = configsPtr + i * 20;
                const componentType =
                    WebGPU.SubgroupMatrixComponentType.indexOf(
                        cfg.componentType);
                const resultComponentType =
                    WebGPU.SubgroupMatrixComponentType.indexOf(
                        cfg.resultComponentType);
                HEAP32[((ptr) >>> 2) >>> 0] = componentType;
                HEAP32[(((ptr) + (4)) >>> 2) >>> 0] = resultComponentType;
                HEAPU32[(((ptr) + (8)) >>> 2) >>> 0] = cfg.M;
                HEAPU32[(((ptr) + (12)) >>> 2) >>> 0] = cfg.N;
                HEAPU32[(((ptr) + (16)) >>> 2) >>> 0] = cfg.K;
              }
            }
            HEAPU32[(((ext) + (12)) >>> 2) >>> 0] = configsPtr;
            HEAPU32[(((ext) + (8)) >>> 2) >>> 0] = configCount;
          }
        });
      },
      AddressMode: [, 'clamp-to-edge', 'repeat', 'mirror-repeat'],
      BlendFactor: [
        , 'zero', 'one', 'src', 'one-minus-src', 'src-alpha',
        'one-minus-src-alpha', 'dst', 'one-minus-dst', 'dst-alpha',
        'one-minus-dst-alpha', 'src-alpha-saturated', 'constant',
        'one-minus-constant', 'src1', 'one-minus-src1', 'src1-alpha',
        'one-minus-src1-alpha'
      ],
      BlendOperation: [, 'add', 'subtract', 'reverse-subtract', 'min', 'max'],
      BufferBindingType: [, , 'uniform', 'storage', 'read-only-storage'],
      BufferMapState: [, 'unmapped', 'pending', 'mapped'],
      CompareFunction: [
        , 'never', 'less', 'equal', 'less-equal', 'greater', 'not-equal',
        'greater-equal', 'always'
      ],
      CompilationInfoRequestStatus: [, 'success', 'callback-cancelled'],
      ComponentSwizzle: [, '0', '1', 'r', 'g', 'b', 'a'],
      CompositeAlphaMode:
          [, 'opaque', 'premultiplied', 'unpremultiplied', 'inherit'],
      CullMode: [, 'none', 'front', 'back'],
      ErrorFilter: [, 'validation', 'out-of-memory', 'internal'],
      FeatureLevel: [, 'compatibility', 'core'],
      FeatureName: {
        1: 'core-features-and-limits',
        2: 'depth-clip-control',
        3: 'depth32float-stencil8',
        4: 'texture-compression-bc',
        5: 'texture-compression-bc-sliced-3d',
        6: 'texture-compression-etc2',
        7: 'texture-compression-astc',
        8: 'texture-compression-astc-sliced-3d',
        9: 'timestamp-query',
        10: 'indirect-first-instance',
        11: 'shader-f16',
        12: 'rg11b10ufloat-renderable',
        13: 'bgra8unorm-storage',
        14: 'float32-filterable',
        15: 'float32-blendable',
        16: 'clip-distances',
        17: 'dual-source-blending',
        18: 'subgroups',
        19: 'texture-formats-tier1',
        20: 'texture-formats-tier2',
        21: 'primitive-index',
        22: 'texture-component-swizzle',
        23: 'subgroup-size-control',
        327692: 'chromium-experimental-unorm16-texture-formats',
        327729: 'chromium-experimental-multi-draw-indirect',
        327732: 'chromium-experimental-subgroup-matrix'
      },
      FilterMode: [, 'nearest', 'linear'],
      FrontFace: [, 'ccw', 'cw'],
      IndexFormat: [, 'uint16', 'uint32'],
      InstanceFeatureName: [
        , 'timed-wait-any', 'shader-source-spirv',
        'multiple-devices-per-adapter'
      ],
      LoadOp: [, 'load', 'clear'],
      MipmapFilterMode: [, 'nearest', 'linear'],
      OptionalBool: ['false', 'true'],
      PowerPreference: [, 'low-power', 'high-performance'],
      PredefinedColorSpace: [, 'srgb', 'display-p3'],
      PrimitiveTopology: [
        , 'point-list', 'line-list', 'line-strip', 'triangle-list',
        'triangle-strip'
      ],
      QueryType: [, 'occlusion', 'timestamp'],
      SamplerBindingType: [, , 'filtering', 'non-filtering', 'comparison'],
      Status: [, 'success', 'error'],
      StencilOperation: [
        , 'keep', 'zero', 'replace', 'invert', 'increment-clamp',
        'decrement-clamp', 'increment-wrap', 'decrement-wrap'
      ],
      StorageTextureAccess: [, , 'write-only', 'read-only', 'read-write'],
      StoreOp: [, 'store', 'discard'],
      SubgroupMatrixComponentType: [, 'f32', 'f16', 'u32', 'i32', 'u8', 'i8'],
      SurfaceGetCurrentTextureStatus: [
        , 'success-optimal', 'success-suboptimal', 'timeout', 'outdated',
        'lost', 'error'
      ],
      TextureAspect: [, 'all', 'stencil-only', 'depth-only'],
      TextureDimension: [, '1d', '2d', '3d'],
      TextureFormat: [
        ,
        'r8unorm',
        'r8snorm',
        'r8uint',
        'r8sint',
        'r16unorm',
        'r16snorm',
        'r16uint',
        'r16sint',
        'r16float',
        'rg8unorm',
        'rg8snorm',
        'rg8uint',
        'rg8sint',
        'r32float',
        'r32uint',
        'r32sint',
        'rg16unorm',
        'rg16snorm',
        'rg16uint',
        'rg16sint',
        'rg16float',
        'rgba8unorm',
        'rgba8unorm-srgb',
        'rgba8snorm',
        'rgba8uint',
        'rgba8sint',
        'bgra8unorm',
        'bgra8unorm-srgb',
        'rgb10a2uint',
        'rgb10a2unorm',
        'rg11b10ufloat',
        'rgb9e5ufloat',
        'rg32float',
        'rg32uint',
        'rg32sint',
        'rgba16unorm',
        'rgba16snorm',
        'rgba16uint',
        'rgba16sint',
        'rgba16float',
        'rgba32float',
        'rgba32uint',
        'rgba32sint',
        'stencil8',
        'depth16unorm',
        'depth24plus',
        'depth24plus-stencil8',
        'depth32float',
        'depth32float-stencil8',
        'bc1-rgba-unorm',
        'bc1-rgba-unorm-srgb',
        'bc2-rgba-unorm',
        'bc2-rgba-unorm-srgb',
        'bc3-rgba-unorm',
        'bc3-rgba-unorm-srgb',
        'bc4-r-unorm',
        'bc4-r-snorm',
        'bc5-rg-unorm',
        'bc5-rg-snorm',
        'bc6h-rgb-ufloat',
        'bc6h-rgb-float',
        'bc7-rgba-unorm',
        'bc7-rgba-unorm-srgb',
        'etc2-rgb8unorm',
        'etc2-rgb8unorm-srgb',
        'etc2-rgb8a1unorm',
        'etc2-rgb8a1unorm-srgb',
        'etc2-rgba8unorm',
        'etc2-rgba8unorm-srgb',
        'eac-r11unorm',
        'eac-r11snorm',
        'eac-rg11unorm',
        'eac-rg11snorm',
        'astc-4x4-unorm',
        'astc-4x4-unorm-srgb',
        'astc-5x4-unorm',
        'astc-5x4-unorm-srgb',
        'astc-5x5-unorm',
        'astc-5x5-unorm-srgb',
        'astc-6x5-unorm',
        'astc-6x5-unorm-srgb',
        'astc-6x6-unorm',
        'astc-6x6-unorm-srgb',
        'astc-8x5-unorm',
        'astc-8x5-unorm-srgb',
        'astc-8x6-unorm',
        'astc-8x6-unorm-srgb',
        'astc-8x8-unorm',
        'astc-8x8-unorm-srgb',
        'astc-10x5-unorm',
        'astc-10x5-unorm-srgb',
        'astc-10x6-unorm',
        'astc-10x6-unorm-srgb',
        'astc-10x8-unorm',
        'astc-10x8-unorm-srgb',
        'astc-10x10-unorm',
        'astc-10x10-unorm-srgb',
        'astc-12x10-unorm',
        'astc-12x10-unorm-srgb',
        'astc-12x12-unorm',
        'astc-12x12-unorm-srgb'
      ],
      TextureSampleType:
          [, , 'float', 'unfilterable-float', 'depth', 'sint', 'uint'],
      TextureViewDimension:
          [, '1d', '2d', '2d-array', 'cube', 'cube-array', '3d'],
      ToneMappingMode: [, 'standard', 'extended'],
      VertexFormat: [
        ,
        'uint8',
        'uint8x2',
        'uint8x4',
        'sint8',
        'sint8x2',
        'sint8x4',
        'unorm8',
        'unorm8x2',
        'unorm8x4',
        'snorm8',
        'snorm8x2',
        'snorm8x4',
        'uint16',
        'uint16x2',
        'uint16x4',
        'sint16',
        'sint16x2',
        'sint16x4',
        'unorm16',
        'unorm16x2',
        'unorm16x4',
        'snorm16',
        'snorm16x2',
        'snorm16x4',
        'float16',
        'float16x2',
        'float16x4',
        'float32',
        'float32x2',
        'float32x3',
        'float32x4',
        'uint32',
        'uint32x2',
        'uint32x3',
        'uint32x4',
        'sint32',
        'sint32x2',
        'sint32x3',
        'sint32x4',
        'unorm10-10-10-2',
        'unorm8x4-bgra',
        'snorm10-10-10-2'
      ],
      VertexStepMode: [, 'vertex', 'instance'],
      WGSLLanguageFeatureName: [
        , 'readonly_and_readwrite_storage_textures',
        'packed_4x8_integer_dot_product', 'unrestricted_pointer_parameters',
        'pointer_composite_access', 'uniform_buffer_standard_layout',
        'subgroup_id', 'texture_and_sampler_let', 'subgroup_uniformity',
        'texture_formats_tier1', 'linear_indexing', 'immediate_address_space',
        'buffer_view', 'swizzle_assignment', 'fragment_depth'
      ]
    };

    function _emscripten_webgpu_get_device() {
      if (WebGPU.preinitializedDeviceId === undefined) {
        WebGPU.preinitializedDeviceId =
            WebGPU.importJsDevice(Module['preinitializedWebGPUDevice']);
        // Some users depend on this keeping the device alive, so we add an
        // additional reference when we first initialize it.
        _wgpuDeviceAddRef(WebGPU.preinitializedDeviceId);
      }
      _wgpuDeviceAddRef(WebGPU.preinitializedDeviceId);
      return WebGPU.preinitializedDeviceId;
    }

    function _emwgpuBufferDestroy(bufferPtr) {
      bufferPtr >>>= 0;
      var buffer = WebGPU.getJsObject(bufferPtr);
      var onUnmap = WebGPU.Internals.bufferOnUnmaps[bufferPtr];
      if (onUnmap) {
        for (var i = 0; i < onUnmap.length; ++i) {
          onUnmap[i]();
        }
        delete WebGPU.Internals.bufferOnUnmaps[bufferPtr];
      }
      buffer.destroy();
    }

    function _emwgpuBufferGetMappedRange(bufferPtr, offset, size) {
      bufferPtr >>>= 0;
      offset >>>= 0;
      size >>>= 0;
      var buffer = WebGPU.getJsObject(bufferPtr);
      if (size == 4294967295) size = undefined;
      var mapped;
      try {
        mapped = buffer.getMappedRange(offset, size);
      } catch (ex) {
        return 0;
      }
      var data = _memalign(16, mapped.byteLength);
      HEAPU8.fill(0, data, mapped.byteLength);
      WebGPU.Internals.bufferOnUnmaps[bufferPtr].push(() => {
        new Uint8Array(mapped).set(
            HEAPU8.subarray(data >>> 0, data + mapped.byteLength >>> 0));
        _free(data);
      });
      return data;
    }

    function _emwgpuBufferUnmap(bufferPtr) {
      bufferPtr >>>= 0;
      var buffer = WebGPU.getJsObject(bufferPtr);
      var onUnmap = WebGPU.Internals.bufferOnUnmaps[bufferPtr];
      if (!onUnmap) {
        // Already unmapped
        return;
      }
      for (var i = 0; i < onUnmap.length; ++i) {
        onUnmap[i]();
      }
      delete WebGPU.Internals.bufferOnUnmaps[bufferPtr];
      buffer.unmap();
    }

    function _emwgpuBufferWriteMappedRange(bufferPtr, offset, data, size) {
      bufferPtr >>>= 0;
      offset >>>= 0;
      data >>>= 0;
      size >>>= 0;
      var buffer = WebGPU.getJsObject(bufferPtr);
      var mapped;
      try {
        mapped = buffer.getMappedRange(offset, size);
      } catch (ex) {
        return 2;
      }
      new Uint8Array(mapped).set(
          HEAPU8.subarray(data >>> 0, data + size >>> 0));
      return 1;
    }

    function _emwgpuDelete(ptr) {
      ptr >>>= 0;
      delete WebGPU.Internals.jsObjects[ptr];
    }

    function _emwgpuDeviceCreateBuffer(devicePtr, descriptor, bufferPtr) {
      devicePtr >>>= 0;
      descriptor >>>= 0;
      bufferPtr >>>= 0;
      var mappedAtCreation = !!(HEAPU32[(((descriptor) + (32)) >>> 2) >>> 0]);
      var desc = {
        'label': WebGPU.makeStringFromOptionalStringView(descriptor + 4),
        'usage': HEAPU32[(((descriptor) + (16)) >>> 2) >>> 0],
        'size': readI53FromI64((descriptor) + (24)),
        'mappedAtCreation': mappedAtCreation
      };
      var device = WebGPU.getJsObject(devicePtr);
      var buffer;
      try {
        buffer = device.createBuffer(desc);
      } catch (ex) {
        // The only exception should be RangeError if mapping at creation ran
        // out of memory.
        return false;
      }
      WebGPU.Internals.jsObjectInsert(bufferPtr, buffer);
      if (mappedAtCreation) {
        WebGPU.Internals.bufferOnUnmaps[bufferPtr] = [];
      }
      return true;
    }

    var _emwgpuDeviceCreateComputePipelineAsync = function(
        devicePtr, futureId_low, futureId_high, descriptor, pipelinePtr) {
      devicePtr >>>= 0;
      var futureId = convertI32PairToI53Checked(futureId_low, futureId_high);
      descriptor >>>= 0;
      pipelinePtr >>>= 0;
      var desc = WebGPU.makeComputePipelineDesc(descriptor);
      var device = WebGPU.getJsObject(devicePtr);
      // createComputePipelineAsync
      WebGPU.Internals.futureInsert(
          futureId,
          device.createComputePipelineAsync(desc).then(
              pipeline => {
                // createComputePipelineAsync fulfilled
                callUserCallback(() => {
                  WebGPU.Internals.jsObjectInsert(pipelinePtr, pipeline);
                  _emwgpuOnCreateComputePipelineCompleted(
                      futureId, 1, pipelinePtr, 0);
                });
              },
              pipelineError => {
                // createComputePipelineAsync rejected
                callUserCallback(() => {
                  var sp = stackSave();
                  var messagePtr = stringToUTF8OnStack(pipelineError.message);
                  var status = pipelineError.reason === 'validation' ? 3 :
                      pipelineError.reason === 'internal'            ? 4 :
                                                                       0;
                  _emwgpuOnCreateComputePipelineCompleted(
                      futureId, status, pipelinePtr, messagePtr);
                  stackRestore(sp);
                });
              }));
    };

    function _emwgpuDeviceCreateShaderModule(
        devicePtr, descriptor, shaderModulePtr) {
      devicePtr >>>= 0;
      descriptor >>>= 0;
      shaderModulePtr >>>= 0;
      var nextInChainPtr = HEAPU32[((descriptor) >>> 2) >>> 0];
      var sType = HEAP32[(((nextInChainPtr) + (4)) >>> 2) >>> 0];
      var desc = {
        'label': WebGPU.makeStringFromOptionalStringView(descriptor + 4),
        'code': ''
      };
      switch (sType) {
        case 2: {
          desc['code'] = WebGPU.makeStringFromStringView(nextInChainPtr + 8);
          break;
        }
      }
      var device = WebGPU.getJsObject(devicePtr);
      WebGPU.Internals.jsObjectInsert(
          shaderModulePtr, device.createShaderModule(desc));
    }

    var _emwgpuDeviceDestroy = devicePtr => {
      const device = WebGPU.getJsObject(devicePtr);
      // Remove the onuncapturederror handler which holds a pointer to the
      // WGPUDevice.
      device.onuncapturederror = null;
      device.destroy();
    };

    var _emwgpuQueueOnSubmittedWorkDone = function(
        queuePtr, futureId_low, futureId_high) {
      queuePtr >>>= 0;
      var futureId = convertI32PairToI53Checked(futureId_low, futureId_high);
      var queue = WebGPU.getJsObject(queuePtr);
      // onSubmittedWorkDone
      WebGPU.Internals.futureInsert(
          futureId, queue.onSubmittedWorkDone().then(() => {
            // onSubmittedWorkDone fulfilled (assumed not to reject)
            callUserCallback(() => {
              _emwgpuOnWorkDoneCompleted(futureId, 1);
            });
          }));
    };

    var _emwgpuWaitAny = function(futurePtr, futureCount, timeoutMSPtr) {
      futurePtr >>>= 0;
      futureCount >>>= 0;
      timeoutMSPtr >>>= 0;
      return Asyncify.handleAsync(async () => {
        var promises = [];
        if (timeoutMSPtr) {
          var timeoutMS = HEAP32[((timeoutMSPtr) >>> 2) >>> 0];
          promises.length = futureCount + 1;
          promises[futureCount] =
              new Promise(resolve => setTimeout(resolve, timeoutMS, 0));
        } else {
          promises.length = futureCount;
        }
        for (var i = 0; i < futureCount; ++i) {
          // If any FutureID is not tracked, it means it must be done.
          var futureId = readI53FromI64((futurePtr + i * 8));
          if (!(futureId in WebGPU.Internals.futures)) {
            return futureId;
          }
          promises[i] = WebGPU.Internals.futures[futureId];
        }
        const firstResolvedFuture = await Promise.race(promises);
        delete WebGPU.Internals.futures[firstResolvedFuture];
        return firstResolvedFuture;
      });
    };

    _emwgpuWaitAny.isAsync = true;

    var ENV = {};

    var getExecutableName = () => thisProgram;

    var getEnvStrings = () => {
      if (!getEnvStrings.strings) {
        // Default values.
        var lang = (globalThis.navigator?.language ?? 'C').replace('-', '_') +
            '.UTF-8';
        var env = {
          'USER': 'web_user',
          'LOGNAME': 'web_user',
          'PATH': '/',
          'PWD': '/',
          'HOME': '/home/web_user',
          'LANG': lang,
          '_': getExecutableName()
        };
        // Apply the user-provided values, if any.
        for (var x in ENV) {
          // x is a key in ENV; if ENV[x] is undefined, that means it was
          // explicitly set to be so. We allow user code to do that to
          // force variables with default values to remain unset.
          if (ENV[x] === undefined)
            delete env[x];
          else
            env[x] = ENV[x];
        }
        var strings = [];
        for (var x in env) {
          strings.push(`${x}=${env[x]}`);
        }
        getEnvStrings.strings = strings;
      }
      return getEnvStrings.strings;
    };

    function _environ_get(__environ, environ_buf) {
      __environ >>>= 0;
      environ_buf >>>= 0;
      var bufSize = 0;
      var envp = 0;
      for (var string of getEnvStrings()) {
        var ptr = environ_buf + bufSize;
        HEAPU32[(((__environ) + (envp)) >>> 2) >>> 0] = ptr;
        bufSize += stringToUTF8(string, ptr, Infinity) + 1;
        envp += 4;
      }
      return 0;
    }

    function _environ_sizes_get(penviron_count, penviron_buf_size) {
      penviron_count >>>= 0;
      penviron_buf_size >>>= 0;
      var strings = getEnvStrings();
      HEAPU32[((penviron_count) >>> 2) >>> 0] = strings.length;
      var bufSize = 0;
      for (var string of strings) {
        bufSize += lengthBytesUTF8(string) + 1;
      }
      HEAPU32[((penviron_buf_size) >>> 2) >>> 0] = bufSize;
      return 0;
    }

    function _fd_close(fd) {
      try {
        var stream = SYSCALLS.getStreamFromFD(fd);
        FS.close(stream);
        return 0;
      } catch (e) {
        if (typeof FS == 'undefined' || !(e.name === 'ErrnoError')) throw e;
        return e.errno;
      }
    }

    /** @param {number=} offset */ var doReadv =
        (stream, iov, iovcnt, offset) => {
          var ret = 0;
          for (var i = 0; i < iovcnt; i++) {
            var ptr = HEAPU32[((iov) >>> 2) >>> 0];
            var len = HEAPU32[(((iov) + (4)) >>> 2) >>> 0];
            iov += 8;
            try {
              var curr = FS.read(stream, HEAP8, ptr, len, offset);
            } catch (e) {
              // On a non-blocking stream a subsequent read may would-block
              // after we already gathered data. POSIX readv is a single
              // gather-read: return what we have rather than failing the whole
              // call.
              if (ret > 0 && e instanceof FS.ErrnoError &&
                  (e.errno == 6 || e.errno == 6)) {
                break;
              }
              throw e;
            }
            if (curr < 0) return -1;
            ret += curr;
            if (curr < len) break;
            // nothing more to read
            if (typeof offset != 'undefined') {
              offset += curr;
            }
          }
          return ret;
        };

    function _fd_pread(fd, iov, iovcnt, offset_low, offset_high, pnum) {
      iov >>>= 0;
      iovcnt >>>= 0;
      var offset = convertI32PairToI53Checked(offset_low, offset_high);
      pnum >>>= 0;
      try {
        if (isNaN(offset)) return 22;
        var stream = SYSCALLS.getStreamFromFD(fd);
        var num = doReadv(stream, iov, iovcnt, offset);
        HEAPU32[((pnum) >>> 2) >>> 0] = num;
        return 0;
      } catch (e) {
        if (typeof FS == 'undefined' || !(e.name === 'ErrnoError')) throw e;
        return e.errno;
      }
    }

    function _fd_read(fd, iov, iovcnt, pnum) {
      iov >>>= 0;
      iovcnt >>>= 0;
      pnum >>>= 0;
      try {
        var stream = SYSCALLS.getStreamFromFD(fd);
        var num = doReadv(stream, iov, iovcnt);
        HEAPU32[((pnum) >>> 2) >>> 0] = num;
        return 0;
      } catch (e) {
        if (typeof FS == 'undefined' || !(e.name === 'ErrnoError')) throw e;
        return e.errno;
      }
    }

    function _fd_seek(fd, offset_low, offset_high, whence, newOffset) {
      var offset = convertI32PairToI53Checked(offset_low, offset_high);
      newOffset >>>= 0;
      try {
        if (isNaN(offset)) return 22;
        var stream = SYSCALLS.getStreamFromFD(fd);
        FS.llseek(stream, offset, whence);
        (tempI64 =
             [
               stream.position >>> 0,
               (tempDouble = stream.position,
                (+(Math.abs(tempDouble))) >= 1 ?
                    (tempDouble > 0 ?
                         (+(Math.floor((tempDouble) / 4294967296))) >>> 0 :
                         (~~((+(Math.ceil(
                             (tempDouble - +(((~~(tempDouble))) >>> 0)) /
                             4294967296))))) >>>
                             0) :
                    0)
             ],
         HEAP32[((newOffset) >>> 2) >>> 0] = tempI64[0],
         HEAP32[(((newOffset) + (4)) >>> 2) >>> 0] = tempI64[1]);
        if (stream.getdents && offset === 0 && whence === 0)
          stream.getdents = null;
        // reset readdir state
        return 0;
      } catch (e) {
        if (typeof FS == 'undefined' || !(e.name === 'ErrnoError')) throw e;
        return e.errno;
      }
    }

    /** @param {number=} offset */ var doWritev =
        (stream, iov, iovcnt, offset) => {
          // Gather all iovecs into one contiguous buffer and issue a single
          // FS.write, matching POSIX writev's single gather-write semantics (as
          // __syscall_sendmsg already does). Per-iovec writes fragment a stream
          // socket send into multiple segments, breaking stream byte semantics.
          if (iovcnt == 1) {
            // Single iovec: write directly from HEAP8, no gather buffer needed.
            return FS.write(
                stream, HEAP8, HEAPU32[((iov) >>> 2) >>> 0],
                HEAPU32[(((iov) + (4)) >>> 2) >>> 0], offset);
          }
          var total = 0;
          for (var i = 0, p = iov; i < iovcnt; i++, p += 8) {
            total += HEAPU32[(((p) + (4)) >>> 2) >>> 0];
          }
          var view = new Uint8Array(total);
          var voff = 0;
          for (var i = 0; i < iovcnt; i++, iov += 8) {
            var ptr = HEAPU32[((iov) >>> 2) >>> 0];
            var len = HEAPU32[(((iov) + (4)) >>> 2) >>> 0];
            view.set(HEAPU8.subarray(ptr >>> 0, ptr + len >>> 0), voff);
            voff += len;
          }
          return FS.write(stream, view, 0, total, offset);
        };

    function _fd_write(fd, iov, iovcnt, pnum) {
      iov >>>= 0;
      iovcnt >>>= 0;
      pnum >>>= 0;
      try {
        var stream = SYSCALLS.getStreamFromFD(fd);
        var num = doWritev(stream, iov, iovcnt);
        HEAPU32[((pnum) >>> 2) >>> 0] = num;
        return 0;
      } catch (e) {
        if (typeof FS == 'undefined' || !(e.name === 'ErrnoError')) throw e;
        return e.errno;
      }
    }

    var _emscripten_glActiveTexture = x0 => GLctx.activeTexture(x0);

    var _glActiveTexture = _emscripten_glActiveTexture;

    var _emscripten_glAttachShader = (program, shader) => {
      GLctx.attachShader(GL.programs[program], GL.shaders[shader]);
    };

    var _glAttachShader = _emscripten_glAttachShader;

    function _emscripten_glBindAttribLocation(program, index, name) {
      name >>>= 0;
      GLctx.bindAttribLocation(GL.programs[program], index, UTF8ToString(name));
    }

    var _glBindAttribLocation = _emscripten_glBindAttribLocation;

    var _emscripten_glBindBuffer = (target, buffer) => {
      // Calling glBindBuffer with an unknown buffer will implicitly create a
      // new one.  Here we bypass `GL.counter` and directly using the ID passed
      // in.
      if (buffer && !GL.buffers[buffer]) {
        var b = GLctx.createBuffer();
        b.name = buffer;
        GL.buffers[buffer] = b;
      }
      if (target == 34962) {
        GLctx.currentArrayBufferBinding = buffer;
      } else if (target == 34963) {
        GLctx.currentElementArrayBufferBinding = buffer;
      }
      if (target == 35051) {
        // In WebGL 2 glReadPixels entry point, we need to use a different WebGL
        // 2 API function call when a buffer is bound to
        // GL_PIXEL_PACK_BUFFER_BINDING point, so must keep track whether that
        // binding point is non-null to know what is the proper API function to
        // call.
        GLctx.currentPixelPackBufferBinding = buffer;
      } else if (target == 35052) {
        // In WebGL 2 gl(Compressed)Tex(Sub)Image[23]D entry points, we need to
        // use a different WebGL 2 API function call when a buffer is bound to
        // GL_PIXEL_UNPACK_BUFFER_BINDING point, so must keep track whether that
        // binding point is non-null to know what is the proper API function to
        // call.
        GLctx.currentPixelUnpackBufferBinding = buffer;
      }
      GLctx.bindBuffer(target, GL.buffers[buffer]);
    };

    var _glBindBuffer = _emscripten_glBindBuffer;

    var _emscripten_glBindFramebuffer = (target, framebuffer) => {
      GLctx.bindFramebuffer(target, GL.framebuffers[framebuffer]);
    };

    var _glBindFramebuffer = _emscripten_glBindFramebuffer;

    var _emscripten_glBindTexture = (target, texture) => {
      GLctx.bindTexture(target, GL.textures[texture]);
    };

    var _glBindTexture = _emscripten_glBindTexture;

    function _emscripten_glBufferData(target, size, data, usage) {
      size >>>= 0;
      data >>>= 0;
      // N.b. here first form specifies a heap subarray, second form an integer
      // size, so the ?: code here is polymorphic. It is advised to avoid
      // randomly mixing both uses in calling code, to avoid any potential JS
      // engine JIT issues.
      GLctx.bufferData(
          target, data ? HEAPU8.subarray(data >>> 0, data + size >>> 0) : size,
          usage);
    }

    var _glBufferData = _emscripten_glBufferData;

    var convertI32PairToI53 = (lo, hi) => (lo >>> 0) + hi * 4294967296;

    function _emscripten_glClientWaitSync(
        sync, flags, timeout_low, timeout_high) {
      sync >>>= 0;
      // WebGL2 vs GLES3 differences: in GLES3, the timeout parameter is a
      // uint64, where 0xFFFFFFFFFFFFFFFFULL means GL_TIMEOUT_IGNORED. In JS,
      // there's no 64-bit value types, so instead timeout is taken to be
      // signed, and GL_TIMEOUT_IGNORED is given value -1. Inherently the value
      // accepted in the timeout is lossy, and can't take in arbitrary u64 bit
      // pattern (but most likely doesn't matter) See
      // https://www.khronos.org/registry/webgl/specs/latest/2.0/#5.15
      var timeout = convertI32PairToI53(timeout_low, timeout_high);
      return GLctx.clientWaitSync(GL.syncs[sync], flags, timeout);
    }

    var _glClientWaitSync = _emscripten_glClientWaitSync;

    var _emscripten_glCompileShader = shader => {
      GLctx.compileShader(GL.shaders[shader]);
    };

    var _glCompileShader = _emscripten_glCompileShader;

    var _emscripten_glCreateProgram = () => {
      var id = GL.getNewId(GL.programs);
      var program = GLctx.createProgram();
      // Store additional information needed for each shader program:
      program.name = id;
      // Lazy cache results of
      // glGetProgramiv(GL_ACTIVE_UNIFORM_MAX_LENGTH/GL_ACTIVE_ATTRIBUTE_MAX_LENGTH/GL_ACTIVE_UNIFORM_BLOCK_MAX_NAME_LENGTH)
      program.maxUniformLength = program.maxAttributeLength =
          program.maxUniformBlockNameLength = 0;
      program.uniformIdCounter = 1;
      GL.programs[id] = program;
      return id;
    };

    var _glCreateProgram = _emscripten_glCreateProgram;

    var _emscripten_glCreateShader = shaderType => {
      var id = GL.getNewId(GL.shaders);
      GL.shaders[id] = GLctx.createShader(shaderType);
      return id;
    };

    var _glCreateShader = _emscripten_glCreateShader;

    function _emscripten_glDeleteFramebuffers(n, framebuffers) {
      framebuffers >>>= 0;
      for (var i = 0; i < n; ++i) {
        var id = HEAP32[(((framebuffers) + (i * 4)) >>> 2) >>> 0];
        var framebuffer = GL.framebuffers[id];
        if (!framebuffer) continue;
        // GL spec: "glDeleteFramebuffers silently ignores 0s and names that do
        // not correspond to existing framebuffer objects".
        GLctx.deleteFramebuffer(framebuffer);
        framebuffer.name = 0;
        GL.framebuffers[id] = null;
      }
    }

    var _glDeleteFramebuffers = _emscripten_glDeleteFramebuffers;

    var _emscripten_glDeleteProgram = id => {
      if (!id) return;
      var program = GL.programs[id];
      if (!program) {
        // glDeleteProgram actually signals an error when deleting a nonexisting
        // object, unlike some other GL delete functions.
        GL.recordError(1281);
        return;
      }
      GLctx.deleteProgram(program);
      program.name = 0;
      GL.programs[id] = null;
    };

    var _glDeleteProgram = _emscripten_glDeleteProgram;

    var _emscripten_glDeleteShader = id => {
      if (!id) return;
      var shader = GL.shaders[id];
      if (!shader) {
        // glDeleteShader actually signals an error when deleting a nonexisting
        // object, unlike some other GL delete functions.
        GL.recordError(1281);
        return;
      }
      GLctx.deleteShader(shader);
      GL.shaders[id] = null;
    };

    var _glDeleteShader = _emscripten_glDeleteShader;

    function _emscripten_glDeleteSync(id) {
      id >>>= 0;
      if (!id) return;
      var sync = GL.syncs[id];
      if (!sync) {
        // glDeleteSync signals an error when deleting a nonexisting object,
        // unlike some other GL delete functions.
        GL.recordError(1281);
        return;
      }
      GLctx.deleteSync(sync);
      sync.name = 0;
      GL.syncs[id] = null;
    }

    var _glDeleteSync = _emscripten_glDeleteSync;

    function _emscripten_glDeleteTextures(n, textures) {
      textures >>>= 0;
      for (var i = 0; i < n; i++) {
        var id = HEAP32[(((textures) + (i * 4)) >>> 2) >>> 0];
        var texture = GL.textures[id];
        // GL spec: "glDeleteTextures silently ignores 0s and names that do not
        // correspond to existing textures".
        if (!texture) continue;
        GLctx.deleteTexture(texture);
        texture.name = 0;
        GL.textures[id] = null;
      }
    }

    var _glDeleteTextures = _emscripten_glDeleteTextures;

    var _emscripten_glDetachShader = (program, shader) => {
      GLctx.detachShader(GL.programs[program], GL.shaders[shader]);
    };

    var _glDetachShader = _emscripten_glDetachShader;

    var _emscripten_glDisableVertexAttribArray = index => {
      var cb = GL.currentContext.clientBuffers[index];
      cb.enabled = false;
      GLctx.disableVertexAttribArray(index);
    };

    var _glDisableVertexAttribArray = _emscripten_glDisableVertexAttribArray;

    var _emscripten_glDrawArrays = (mode, first, count) => {
      // bind any client-side buffers
      GL.preDrawHandleClientVertexAttribBindings(first + count);
      GLctx.drawArrays(mode, first, count);
      GL.postDrawHandleClientVertexAttribBindings();
    };

    var _glDrawArrays = _emscripten_glDrawArrays;

    var _emscripten_glEnableVertexAttribArray = index => {
      var cb = GL.currentContext.clientBuffers[index];
      cb.enabled = true;
      GLctx.enableVertexAttribArray(index);
    };

    var _glEnableVertexAttribArray = _emscripten_glEnableVertexAttribArray;

    function _emscripten_glFenceSync(condition, flags) {
      var sync = GLctx.fenceSync(condition, flags);
      if (sync) {
        var id = GL.getNewId(GL.syncs);
        sync.name = id;
        GL.syncs[id] = sync;
        return id;
      }
      return 0;
    }

    var _glFenceSync = _emscripten_glFenceSync;

    var _emscripten_glFinish = () => GLctx.finish();

    var _glFinish = _emscripten_glFinish;

    var _emscripten_glFramebufferTexture2D =
        (target, attachment, textarget, texture, level) => {
          GLctx.framebufferTexture2D(
              target, attachment, textarget, GL.textures[texture], level);
        };

    var _glFramebufferTexture2D = _emscripten_glFramebufferTexture2D;

    function _emscripten_glGenBuffers(n, buffers) {
      buffers >>>= 0;
      GL.genObject(n, buffers, 'createBuffer', GL.buffers);
    }

    var _glGenBuffers = _emscripten_glGenBuffers;

    function _emscripten_glGenFramebuffers(n, ids) {
      ids >>>= 0;
      GL.genObject(n, ids, 'createFramebuffer', GL.framebuffers);
    }

    var _glGenFramebuffers = _emscripten_glGenFramebuffers;

    function _emscripten_glGenTextures(n, textures) {
      textures >>>= 0;
      GL.genObject(n, textures, 'createTexture', GL.textures);
    }

    var _glGenTextures = _emscripten_glGenTextures;

    var _emscripten_glGetError = () => {
      var error = GLctx.getError() || GL.lastError;
      GL.lastError = 0;
      return error;
    };

    var _glGetError = _emscripten_glGetError;

    var webglGetExtensions = () => {
      var exts = getEmscriptenSupportedExtensions(GLctx);
      exts = exts.concat(exts.map(e => 'GL_' + e));
      return exts;
    };

    var emscriptenWebGLGet = (name_, p, type) => {
      // Guard against user passing a null pointer.
      // Note that GLES2 spec does not say anything about how passing a null
      // pointer should be treated.  Testing on desktop core GL 3, the
      // application crashes on glGetIntegerv to a null pointer, but better to
      // report an error instead of doing anything random.
      if (!p) {
        GL.recordError(1281);
        return;
      }
      var ret = undefined;
      switch (name_) {
          // Handle a few trivial GLES values
        case 36346:
          // GL_SHADER_COMPILER
          ret = 1;
          break;

        case 36344:
          // GL_SHADER_BINARY_FORMATS
          if (type != 0 && type != 1) {
            GL.recordError(1280);
          }
          // Do not write anything to the out pointer, since no binary formats
          // are supported.
          return;

        case 34814:
          // GL_NUM_PROGRAM_BINARY_FORMATS
        case 36345:
          // GL_NUM_SHADER_BINARY_FORMATS
          ret = 0;
          break;

        case 34466:
          // GL_NUM_COMPRESSED_TEXTURE_FORMATS
          // WebGL doesn't have GL_NUM_COMPRESSED_TEXTURE_FORMATS (it's obsolete
          // since GL_COMPRESSED_TEXTURE_FORMATS returns a JS array that can be
          // queried for length), so implement it ourselves to allow C++ GLES2
          // code to get the length.
          var formats = GLctx.getParameter(34467);
          ret = formats ? formats.length : 0;
          break;

        case 33309:
          // GL_NUM_EXTENSIONS
          if (GL.currentContext.version < 2) {
            // Calling GLES3/WebGL2 function with a GLES2/WebGL1 context
            GL.recordError(1282);
            return;
          }
          ret = webglGetExtensions().length;
          break;

        case 33307:
          // GL_MAJOR_VERSION
        case 33308:
          // GL_MINOR_VERSION
          if (GL.currentContext.version < 2) {
            GL.recordError(1280);
            // GL_INVALID_ENUM
            return;
          }
          ret = name_ == 33307 ? 3 : 0;
          // return version 3.0
          break;
      }
      if (ret === undefined) {
        var result = GLctx.getParameter(name_);
        switch (typeof result) {
          case 'number':
            ret = result;
            break;

          case 'boolean':
            ret = result ? 1 : 0;
            break;

          case 'string':
            GL.recordError(1280);
            // GL_INVALID_ENUM
            return;

          case 'object':
            if (result === null) {
              // null is a valid result for some (e.g., which buffer is bound -
              // perhaps nothing is bound), but otherwise can mean an invalid
              // name_, which we need to report as an error
              switch (name_) {
                case 34964:
                  // ARRAY_BUFFER_BINDING
                case 35725:
                  // CURRENT_PROGRAM
                case 34965:
                  // ELEMENT_ARRAY_BUFFER_BINDING
                case 36006:
                  // FRAMEBUFFER_BINDING or DRAW_FRAMEBUFFER_BINDING
                case 36007:
                  // RENDERBUFFER_BINDING
                case 32873:
                  // TEXTURE_BINDING_2D
                case 34229:
                  // WebGL 2 GL_VERTEX_ARRAY_BINDING, or WebGL 1 extension
                  // OES_vertex_array_object GL_VERTEX_ARRAY_BINDING_OES
                case 36662:
                  // COPY_READ_BUFFER_BINDING or COPY_READ_BUFFER
                case 36663:
                  // COPY_WRITE_BUFFER_BINDING or COPY_WRITE_BUFFER
                case 35053:
                  // PIXEL_PACK_BUFFER_BINDING
                case 35055:
                  // PIXEL_UNPACK_BUFFER_BINDING
                case 36010:
                  // READ_FRAMEBUFFER_BINDING
                case 35097:
                  // SAMPLER_BINDING
                case 35869:
                  // TEXTURE_BINDING_2D_ARRAY
                case 32874:
                  // TEXTURE_BINDING_3D
                case 36389:
                  // TRANSFORM_FEEDBACK_BINDING
                case 35983:
                  // TRANSFORM_FEEDBACK_BUFFER_BINDING
                case 35368:
                  // UNIFORM_BUFFER_BINDING
                case 34068: {
                  // TEXTURE_BINDING_CUBE_MAP
                  ret = 0;
                  break;
                }

                default: {
                  GL.recordError(1280);
                  // GL_INVALID_ENUM
                  return;
                }
              }
            } else if (
                result instanceof Float32Array ||
                result instanceof Uint32Array || result instanceof Int32Array ||
                result instanceof Array) {
              for (var i = 0; i < result.length; ++i) {
                switch (type) {
                  case 0:
                    HEAP32[(((p) + (i * 4)) >>> 2) >>> 0] = result[i];
                    break;

                  case 2:
                    HEAPF32[(((p) + (i * 4)) >>> 2) >>> 0] = result[i];
                    break;

                  case 4:
                    HEAP8[(p) + (i) >>> 0] = result[i] ? 1 : 0;
                    break;
                }
              }
              return;
            } else {
              try {
                ret = result.name | 0;
              } catch (e) {
                GL.recordError(1280);
                // GL_INVALID_ENUM
                err(`GL_INVALID_ENUM in glGet${
                    type}v: Unknown object returned from WebGL getParameter(${
                    name_})! (error: ${e})`);
                return;
              }
            }
            break;

          default:
            GL.recordError(1280);
            // GL_INVALID_ENUM
            err(`GL_INVALID_ENUM in glGet${type}v: Native code calling glGet${
                type}v(${name_}) and it returns ${result} of type ${
                typeof (result)}!`);
            return;
        }
      }
      switch (type) {
        case 1:
          writeI53ToI64(p, ret);
          break;

        case 0:
          HEAP32[((p) >>> 2) >>> 0] = ret;
          break;

        case 2:
          HEAPF32[((p) >>> 2) >>> 0] = ret;
          break;

        case 4:
          HEAP8[p >>> 0] = ret ? 1 : 0;
          break;
      }
    };

    function _emscripten_glGetIntegerv(name_, p) {
      p >>>= 0;
      return emscriptenWebGLGet(name_, p, 0);
    }

    var _glGetIntegerv = _emscripten_glGetIntegerv;

    function _emscripten_glGetProgramiv(program, pname, p) {
      p >>>= 0;
      if (!p) {
        // GLES2 specification does not specify how to behave if p is a null
        // pointer. Since calling this function does not make sense if p ==
        // null, issue a GL error to notify user about it.
        GL.recordError(1281);
        return;
      }
      if (program >= GL.counter) {
        GL.recordError(1281);
        return;
      }
      program = GL.programs[program];
      if (pname == 35716) {
        // GL_INFO_LOG_LENGTH
        var log = GLctx.getProgramInfoLog(program);
        if (log === null) log = '(unknown error)';
        HEAP32[((p) >>> 2) >>> 0] = log.length + 1;
      } else if (pname == 35719) {
        if (!program.maxUniformLength) {
          var numActiveUniforms = GLctx.getProgramParameter(program, 35718);
          for (var i = 0; i < numActiveUniforms; ++i) {
            program.maxUniformLength = Math.max(
                program.maxUniformLength,
                GLctx.getActiveUniform(program, i).name.length + 1);
          }
        }
        HEAP32[((p) >>> 2) >>> 0] = program.maxUniformLength;
      } else if (pname == 35722) {
        if (!program.maxAttributeLength) {
          var numActiveAttributes = GLctx.getProgramParameter(program, 35721);
          for (var i = 0; i < numActiveAttributes; ++i) {
            program.maxAttributeLength = Math.max(
                program.maxAttributeLength,
                GLctx.getActiveAttrib(program, i).name.length + 1);
          }
        }
        HEAP32[((p) >>> 2) >>> 0] = program.maxAttributeLength;
      } else if (pname == 35381) {
        if (!program.maxUniformBlockNameLength) {
          var numActiveUniformBlocks =
              GLctx.getProgramParameter(program, 35382);
          for (var i = 0; i < numActiveUniformBlocks; ++i) {
            program.maxUniformBlockNameLength = Math.max(
                program.maxUniformBlockNameLength,
                GLctx.getActiveUniformBlockName(program, i).length + 1);
          }
        }
        HEAP32[((p) >>> 2) >>> 0] = program.maxUniformBlockNameLength;
      } else {
        HEAP32[((p) >>> 2) >>> 0] = GLctx.getProgramParameter(program, pname);
      }
    }

    var _glGetProgramiv = _emscripten_glGetProgramiv;

    function _emscripten_glGetShaderInfoLog(
        shader, maxLength, length, infoLog) {
      length >>>= 0;
      infoLog >>>= 0;
      var log = GLctx.getShaderInfoLog(GL.shaders[shader]);
      if (log === null) log = '(unknown error)';
      var numBytesWrittenExclNull = (maxLength > 0 && infoLog) ?
          stringToUTF8(log, infoLog, maxLength) :
          0;
      if (length) HEAP32[((length) >>> 2) >>> 0] = numBytesWrittenExclNull;
    }

    var _glGetShaderInfoLog = _emscripten_glGetShaderInfoLog;

    function _emscripten_glGetShaderiv(shader, pname, p) {
      p >>>= 0;
      if (!p) {
        // GLES2 specification does not specify how to behave if p is a null
        // pointer. Since calling this function does not make sense if p ==
        // null, issue a GL error to notify user about it.
        GL.recordError(1281);
        return;
      }
      if (pname == 35716) {
        // GL_INFO_LOG_LENGTH
        var log = GLctx.getShaderInfoLog(GL.shaders[shader]);
        if (log === null) log = '(unknown error)';
        // The GLES2 specification says that if the shader has an empty info
        // log, a value of 0 is returned. Otherwise the log has a null char
        // appended. (An empty string is falsey, so we can just check that
        // instead of looking at log.length.)
        var logLength = log ? log.length + 1 : 0;
        HEAP32[((p) >>> 2) >>> 0] = logLength;
      } else if (pname == 35720) {
        // GL_SHADER_SOURCE_LENGTH
        var source = GLctx.getShaderSource(GL.shaders[shader]);
        // source may be a null, or the empty string, both of which are falsey
        // values that we report a 0 length for.
        var sourceLength = source ? source.length + 1 : 0;
        HEAP32[((p) >>> 2) >>> 0] = sourceLength;
      } else {
        HEAP32[((p) >>> 2) >>> 0] =
            GLctx.getShaderParameter(GL.shaders[shader], pname);
      }
    }

    var _glGetShaderiv = _emscripten_glGetShaderiv;

    function _emscripten_glGetString(name_) {
      var ret = GL.stringCache[name_];
      if (!ret) {
        switch (name_) {
          case 7939:
            ret = stringToNewUTF8(webglGetExtensions().join(' '));
            break;

          case 7936:
          case 7937:
          case 37445:
          case 37446:
            var s = GLctx.getParameter(name_);
            if (!s) {
              GL.recordError(1280);
            }
            ret = s ? stringToNewUTF8(s) : 0;
            break;

          case 7938:
            var webGLVersion = GLctx.getParameter(7938);
            // return GLES version string corresponding to the version of the
            // WebGL context
            var glVersion = `OpenGL ES 2.0 (${webGLVersion})`;
            if (GL.currentContext.version >= 2)
              glVersion = `OpenGL ES 3.0 (${webGLVersion})`;
            ret = stringToNewUTF8(glVersion);
            break;

          case 35724:
            var glslVersion = GLctx.getParameter(35724);
            // extract the version number 'N.M' from the string 'WebGL GLSL ES
            // N.M ...'
            var ver_re = /^WebGL GLSL ES ([0-9]\.[0-9][0-9]?)(?:$| .*)/;
            var ver_num = glslVersion.match(ver_re);
            if (ver_num !== null) {
              if (ver_num[1].length == 3) ver_num[1] = ver_num[1] + '0';
              // ensure minor version has 2 digits
              glslVersion = `OpenGL ES GLSL ES ${ver_num[1]} (${glslVersion})`;
            }
            ret = stringToNewUTF8(glslVersion);
            break;

          default:
            GL.recordError(1280);
        }
        GL.stringCache[name_] = ret;
      }
      return ret;
    }

    var _glGetString = _emscripten_glGetString;

    /** @suppress {checkTypes} */ var jstoi_q = str => parseInt(str);

    /** @noinline */ var webglGetLeftBracePos = name =>
        name.slice(-1) == ']' && name.lastIndexOf('[');

    var webglPrepareUniformLocationsBeforeFirstUse = program => {
      var uniformLocsById =
              program.uniformLocsById,  // Maps GLuint -> WebGLUniformLocation
          uniformSizeAndIdsByName =
              program.uniformSizeAndIdsByName,  // Maps name -> [uniform array
                                                // length, GLuint]
          i, j;
      // On the first time invocation of glGetUniformLocation on this shader
      // program: initialize cache data structures and discover which uniforms
      // are arrays.
      if (!uniformLocsById) {
        // maps GLint integer locations to WebGLUniformLocations
        program.uniformLocsById = uniformLocsById = {};
        // maps integer locations back to uniform name strings, so that we can
        // lazily fetch uniform array locations
        program.uniformArrayNamesById = {};
        var numActiveUniforms = GLctx.getProgramParameter(program, 35718);
        for (i = 0; i < numActiveUniforms; ++i) {
          var u = GLctx.getActiveUniform(program, i);
          var nm = u.name;
          var sz = u.size;
          var lb = webglGetLeftBracePos(nm);
          var arrayName = lb > 0 ? nm.slice(0, lb) : nm;
          // Assign a new location.
          var id = program.uniformIdCounter;
          program.uniformIdCounter += sz;
          // Eagerly get the location of the uniformArray[0] base element.
          // The remaining indices >0 will be left for lazy evaluation to
          // improve performance. Those may never be needed to fetch, if the
          // application fills arrays always in full starting from the first
          // element of the array.
          uniformSizeAndIdsByName[arrayName] = [sz, id];
          // Store placeholder integers in place that highlight that these
          // >0 index locations are array indices pending population.
          for (j = 0; j < sz; ++j) {
            uniformLocsById[id] = j;
            program.uniformArrayNamesById[id++] = arrayName;
          }
        }
      }
    };

    function _emscripten_glGetUniformLocation(program, name) {
      name >>>= 0;
      name = UTF8ToString(name);
      if (program = GL.programs[program]) {
        webglPrepareUniformLocationsBeforeFirstUse(program);
        var uniformLocsById = program.uniformLocsById;
        // Maps GLuint -> WebGLUniformLocation
        var arrayIndex = 0;
        var uniformBaseName = name;
        // Invariant: when populating integer IDs for uniform locations, we must
        // maintain the precondition that arrays reside in contiguous addresses,
        // i.e. for a 'vec4 colors[10];', colors[4] must be at location
        // colors[0]+4.  However, user might call glGetUniformLocation(program,
        // "colors") for an array, so we cannot discover based on the user input
        // arguments whether the uniform we are dealing with is an array. The
        // only way to discover which uniforms are arrays is to enumerate over
        // all the active uniforms in the program.
        var leftBrace = webglGetLeftBracePos(name);
        // If user passed an array accessor "[index]", parse the array index off
        // the accessor.
        if (leftBrace > 0) {
          arrayIndex = jstoi_q(name.slice(leftBrace + 1)) >>> 0;
          // "index]", coerce parseInt(']') with >>>0 to treat "foo[]" as
          // "foo[0]" and foo[-1] as unsigned out-of-bounds.
          uniformBaseName = name.slice(0, leftBrace);
        }
        // Have we cached the location of this uniform before?
        // A pair [array length, GLint of the uniform location]
        var sizeAndId = program.uniformSizeAndIdsByName[uniformBaseName];
        // If a uniform with this name exists, and if its index is within the
        // array limits (if it's even an array), query the WebGLlocation, or
        // return an existing cached location.
        if (sizeAndId && arrayIndex < sizeAndId[0]) {
          arrayIndex += sizeAndId[1];
          // Add the base location of the uniform to the array index offset.
          if ((uniformLocsById[arrayIndex] = uniformLocsById[arrayIndex] ||
                   GLctx.getUniformLocation(program, name))) {
            return arrayIndex;
          }
        }
      } else {
        // N.b. we are currently unable to distinguish between GL program IDs
        // that never existed vs GL program IDs that have been deleted, so
        // report GL_INVALID_VALUE in both cases.
        GL.recordError(1281);
      }
      return -1;
    }

    var _glGetUniformLocation = _emscripten_glGetUniformLocation;

    var _emscripten_glIsTexture = id => {
      var texture = GL.textures[id];
      if (!texture) return 0;
      return GLctx.isTexture(texture);
    };

    var _glIsTexture = _emscripten_glIsTexture;

    var _emscripten_glLinkProgram = program => {
      program = GL.programs[program];
      GLctx.linkProgram(program);
      // Invalidate earlier computed uniform->ID mappings, those have now become
      // stale
      program.uniformLocsById = 0;
      // Mark as null-like so that glGetUniformLocation() knows to populate this
      // again.
      program.uniformSizeAndIdsByName = {};
    };

    var _glLinkProgram = _emscripten_glLinkProgram;

    var _emscripten_glPixelStorei = (pname, param) => {
      if (pname == 3317) {
        GL.unpackAlignment = param;
      } else if (pname == 3314) {
        GL.unpackRowLength = param;
      }
      GLctx.pixelStorei(pname, param);
    };

    var _glPixelStorei = _emscripten_glPixelStorei;

    var computeUnpackAlignedImageSize = (width, height, sizePerPixel) => {
      function roundedToNextMultipleOf(x, y) {
        return (x + y - 1) & -y;
      }
      var plainRowSize = (GL.unpackRowLength || width) * sizePerPixel;
      var alignedRowSize =
          roundedToNextMultipleOf(plainRowSize, GL.unpackAlignment);
      return height * alignedRowSize;
    };

    var colorChannelsInGlTextureFormat = format => {
      // Micro-optimizations for size: map format to size by subtracting
      // smallest enum value (0x1902) from all values first.  Also omit the most
      // common size value (1) from the list, which is assumed by formats not on
      // the list.
      var colorChannels = {
        // 0x1902 /* GL_DEPTH_COMPONENT */ - 0x1902: 1,
        // 0x1906 /* GL_ALPHA */ - 0x1902: 1,
        5: 3,
        6: 4,
        // 0x1909 /* GL_LUMINANCE */ - 0x1902: 1,
        8: 2,
        29502: 3,
        29504: 4,
        // 0x1903 /* GL_RED */ - 0x1902: 1,
        26917: 2,
        26918: 2,
        // 0x8D94 /* GL_RED_INTEGER */ - 0x1902: 1,
        29846: 3,
        29847: 4
      };
      return colorChannels[format - 6402] || 1;
    };

    var heapObjectForWebGLType = type => {
      // Micro-optimization for size: Subtract lowest GL enum number (0x1400/*
      // GL_BYTE */) from type to compare smaller values for the heap, for
      // shorter generated code size. Also the type HEAPU16 is not tested for
      // explicitly, but any unrecognized type will return out HEAPU16. (since
      // most types are HEAPU16)
      type -= 5120;
      if (type == 0) return HEAP8;
      if (type == 1) return HEAPU8;
      if (type == 2) return HEAP16;
      if (type == 4) return HEAP32;
      if (type == 6) return HEAPF32;
      if (type == 5 || type == 28922 || type == 28520 || type == 30779 ||
          type == 30782)
        return HEAPU32;
      return HEAPU16;
    };

    var toTypedArrayIndex = (pointer, heap) =>
        pointer >>> (31 - Math.clz32(heap.BYTES_PER_ELEMENT));

    var emscriptenWebGLGetTexPixelData =
        (type, format, width, height, pixels) => {
          var heap = heapObjectForWebGLType(type);
          var sizePerPixel =
              colorChannelsInGlTextureFormat(format) * heap.BYTES_PER_ELEMENT;
          var bytes =
              computeUnpackAlignedImageSize(width, height, sizePerPixel);
          return heap.subarray(
              toTypedArrayIndex(pixels, heap) >>> 0,
              toTypedArrayIndex(pixels + bytes, heap) >>> 0);
        };

    function _emscripten_glReadPixels(
        x, y, width, height, format, type, pixels) {
      pixels >>>= 0;
      if (GL.currentContext.version >= 2) {
        if (GLctx.currentPixelPackBufferBinding) {
          GLctx.readPixels(x, y, width, height, format, type, pixels);
          return;
        }
      }
      var pixelData =
          emscriptenWebGLGetTexPixelData(type, format, width, height, pixels);
      if (!pixelData) {
        GL.recordError(1280);
        return;
      }
      GLctx.readPixels(x, y, width, height, format, type, pixelData);
    }

    var _glReadPixels = _emscripten_glReadPixels;

    function _emscripten_glShaderSource(shader, count, string, length) {
      string >>>= 0;
      length >>>= 0;
      var source = GL.getSource(shader, count, string, length);
      GLctx.shaderSource(GL.shaders[shader], source);
    }

    var _glShaderSource = _emscripten_glShaderSource;

    function _emscripten_glTexImage2D(
        target, level, internalFormat, width, height, border, format, type,
        pixels) {
      pixels >>>= 0;
      if (GL.currentContext.version >= 2) {
        if (GLctx.currentPixelUnpackBufferBinding) {
          GLctx.texImage2D(
              target, level, internalFormat, width, height, border, format,
              type, pixels);
          return;
        }
      }
      var pixelData = pixels ?
          emscriptenWebGLGetTexPixelData(type, format, width, height, pixels) :
          null;
      GLctx.texImage2D(
          target, level, internalFormat, width, height, border, format, type,
          pixelData);
    }

    var _glTexImage2D = _emscripten_glTexImage2D;

    var _emscripten_glTexParameteri = (x0, x1, x2) =>
        GLctx.texParameteri(x0, x1, x2);

    var _glTexParameteri = _emscripten_glTexParameteri;

    var _emscripten_glTexStorage2D = (x0, x1, x2, x3, x4) =>
        GLctx.texStorage2D(x0, x1, x2, x3, x4);

    var _glTexStorage2D = _emscripten_glTexStorage2D;

    var webglGetProgramUniformLocation = (program, location) => {
      if (program) {
        var webglLoc = program.uniformLocsById[location];
        // program.uniformLocsById[location] stores either an integer, or a
        // WebGLUniformLocation.
        // If an integer, we have not yet bound the location, so do it now. The
        // integer value specifies the array index we should bind to.
        if (typeof webglLoc == 'number') {
          program.uniformLocsById[location] = webglLoc =
              GLctx.getUniformLocation(
                  program,
                  program.uniformArrayNamesById[location] +
                      (webglLoc > 0 ? `[${webglLoc}]` : ''));
        }
        // Else an already cached WebGLUniformLocation, return it.
        return webglLoc;
      } else {
        GL.recordError(1282);
      }
    };

    var webglGetUniformLocation = location =>
        webglGetProgramUniformLocation(GLctx.currentProgram, location);

    var _emscripten_glUniform1i = (location, v0) => {
      GLctx.uniform1i(webglGetUniformLocation(location), v0);
    };

    var _glUniform1i = _emscripten_glUniform1i;

    var _emscripten_glUseProgram = program => {
      program = GL.programs[program];
      GLctx.useProgram(program);
      // Record the currently active program so that we can access the uniform
      // mapping table of that program.
      GLctx.currentProgram = program;
    };

    var _glUseProgram = _emscripten_glUseProgram;

    function _emscripten_glVertexAttribPointer(
        index, size, type, normalized, stride, ptr) {
      ptr >>>= 0;
      var cb = GL.currentContext.clientBuffers[index];
      if (!GLctx.currentArrayBufferBinding) {
        cb.size = size;
        cb.type = type;
        cb.normalized = normalized;
        cb.stride = stride;
        cb.ptr = ptr;
        cb.clientside = true;
        cb.vertexAttribPointerAdaptor = /** @this {WebGLRenderingContext} */
            function(index, size, type, normalized, stride, ptr) {
          this.vertexAttribPointer(index, size, type, normalized, stride, ptr);
        };
        return;
      }
      cb.clientside = false;
      GLctx.vertexAttribPointer(index, size, type, !!normalized, stride, ptr);
    }

    var _glVertexAttribPointer = _emscripten_glVertexAttribPointer;

    var _emscripten_glViewport = (x0, x1, x2, x3) =>
        GLctx.viewport(x0, x1, x2, x3);

    var _glViewport = _emscripten_glViewport;

    function _random_get(buffer, size) {
      buffer >>>= 0;
      size >>>= 0;
      return randomFill(HEAPU8.subarray(buffer >>> 0, buffer + size >>> 0));
    }

    var _wgpuBufferGetSize = function(bufferPtr) {
      bufferPtr >>>= 0;
      var ret = (() => {
        var buffer = WebGPU.getJsObject(bufferPtr);
        // 64-bit
        return buffer.size;
      })();
      return (
          setTempRet0(
              (tempDouble = ret,
               (+(Math.abs(tempDouble))) >= 1 ?
                   (tempDouble > 0 ?
                        (+(Math.floor((tempDouble) / 4294967296))) >>> 0 :
                        (~~((+(Math.ceil(
                            (tempDouble - +(((~~(tempDouble))) >>> 0)) /
                            4294967296))))) >>>
                            0) :
                   0)),
          ret >>> 0);
    };

    var _wgpuBufferGetUsage = function(bufferPtr) {
      bufferPtr >>>= 0;
      var ret = (() => {
        var buffer = WebGPU.getJsObject(bufferPtr);
        return buffer.usage;
      })();
      return (
          setTempRet0(
              (tempDouble = ret,
               (+(Math.abs(tempDouble))) >= 1 ?
                   (tempDouble > 0 ?
                        (+(Math.floor((tempDouble) / 4294967296))) >>> 0 :
                        (~~((+(Math.ceil(
                            (tempDouble - +(((~~(tempDouble))) >>> 0)) /
                            4294967296))))) >>>
                            0) :
                   0)),
          ret >>> 0);
    };

    function _wgpuCommandEncoderBeginComputePass(encoderPtr, descriptor) {
      encoderPtr >>>= 0;
      descriptor >>>= 0;
      var desc;
      if (descriptor) {
        desc = {
          'label': WebGPU.makeStringFromOptionalStringView(descriptor + 4),
          'timestampWrites': WebGPU.makePassTimestampWrites(
              HEAPU32[(((descriptor) + (12)) >>> 2) >>> 0])
        };
      }
      var commandEncoder = WebGPU.getJsObject(encoderPtr);
      var ptr = _emwgpuCreateComputePassEncoder(0);
      WebGPU.Internals.jsObjectInsert(
          ptr, commandEncoder.beginComputePass(desc));
      return ptr;
    }

    function _wgpuCommandEncoderCopyBufferToBuffer(
        encoderPtr, srcPtr, srcOffset_low, srcOffset_high, dstPtr,
        dstOffset_low, dstOffset_high, size_low, size_high) {
      encoderPtr >>>= 0;
      srcPtr >>>= 0;
      var srcOffset = convertI32PairToI53Checked(srcOffset_low, srcOffset_high);
      dstPtr >>>= 0;
      var dstOffset = convertI32PairToI53Checked(dstOffset_low, dstOffset_high);
      var size = convertI32PairToI53Checked(size_low, size_high);
      var commandEncoder = WebGPU.getJsObject(encoderPtr);
      var src = WebGPU.getJsObject(srcPtr);
      var dst = WebGPU.getJsObject(dstPtr);
      commandEncoder.copyBufferToBuffer(src, srcOffset, dst, dstOffset, size);
    }

    function _wgpuCommandEncoderCopyTextureToBuffer(
        encoderPtr, srcPtr, dstPtr, copySizePtr) {
      encoderPtr >>>= 0;
      srcPtr >>>= 0;
      dstPtr >>>= 0;
      copySizePtr >>>= 0;
      var commandEncoder = WebGPU.getJsObject(encoderPtr);
      var copySize = WebGPU.makeExtent3D(copySizePtr);
      commandEncoder.copyTextureToBuffer(
          WebGPU.makeTexelCopyTextureInfo(srcPtr),
          WebGPU.makeTexelCopyBufferInfo(dstPtr), copySize);
    }

    function _wgpuCommandEncoderCopyTextureToTexture(
        encoderPtr, srcPtr, dstPtr, copySizePtr) {
      encoderPtr >>>= 0;
      srcPtr >>>= 0;
      dstPtr >>>= 0;
      copySizePtr >>>= 0;
      var commandEncoder = WebGPU.getJsObject(encoderPtr);
      var copySize = WebGPU.makeExtent3D(copySizePtr);
      commandEncoder.copyTextureToTexture(
          WebGPU.makeTexelCopyTextureInfo(srcPtr),
          WebGPU.makeTexelCopyTextureInfo(dstPtr), copySize);
    }

    function _wgpuCommandEncoderFinish(encoderPtr, descriptor) {
      encoderPtr >>>= 0;
      descriptor >>>= 0;
      // TODO: Use the descriptor.
      var commandEncoder = WebGPU.getJsObject(encoderPtr);
      var ptr = _emwgpuCreateCommandBuffer(0);
      WebGPU.Internals.jsObjectInsert(ptr, commandEncoder.finish());
      return ptr;
    }

    function _wgpuCommandEncoderResolveQuerySet(
        encoderPtr, querySetPtr, firstQuery, queryCount, destinationPtr,
        destinationOffset_low, destinationOffset_high) {
      encoderPtr >>>= 0;
      querySetPtr >>>= 0;
      destinationPtr >>>= 0;
      var destinationOffset = convertI32PairToI53Checked(
          destinationOffset_low, destinationOffset_high);
      var commandEncoder = WebGPU.getJsObject(encoderPtr);
      var querySet = WebGPU.getJsObject(querySetPtr);
      var destination = WebGPU.getJsObject(destinationPtr);
      commandEncoder.resolveQuerySet(
          querySet, firstQuery, queryCount, destination, destinationOffset);
    }

    function _wgpuComputePassEncoderDispatchWorkgroups(passPtr, x, y, z) {
      passPtr >>>= 0;
      var pass = WebGPU.getJsObject(passPtr);
      pass.dispatchWorkgroups(x, y, z);
    }

    function _wgpuComputePassEncoderEnd(passPtr) {
      passPtr >>>= 0;
      var pass = WebGPU.getJsObject(passPtr);
      pass.end();
    }

    function _wgpuComputePassEncoderSetBindGroup(
        passPtr, groupIndex, groupPtr, dynamicOffsetCount, dynamicOffsetsPtr) {
      passPtr >>>= 0;
      groupPtr >>>= 0;
      dynamicOffsetCount >>>= 0;
      dynamicOffsetsPtr >>>= 0;
      var pass = WebGPU.getJsObject(passPtr);
      var group = WebGPU.getJsObject(groupPtr);
      if (dynamicOffsetCount == 0) {
        pass.setBindGroup(groupIndex, group);
      } else {
        pass.setBindGroup(
            groupIndex, group, HEAPU32, ((dynamicOffsetsPtr) >>> 2),
            dynamicOffsetCount);
      }
    }

    function _wgpuComputePassEncoderSetPipeline(passPtr, pipelinePtr) {
      passPtr >>>= 0;
      pipelinePtr >>>= 0;
      var pass = WebGPU.getJsObject(passPtr);
      var pipeline = WebGPU.getJsObject(pipelinePtr);
      pass.setPipeline(pipeline);
    }

    var _wgpuDeviceCreateBindGroup = function(devicePtr, descriptor) {
      devicePtr >>>= 0;
      descriptor >>>= 0;
      function makeEntry(entryPtr) {
        var bufferPtr = HEAPU32[(((entryPtr) + (8)) >>> 2) >>> 0];
        var samplerPtr = HEAPU32[(((entryPtr) + (32)) >>> 2) >>> 0];
        var textureViewPtr = HEAPU32[(((entryPtr) + (36)) >>> 2) >>> 0];
        var externalTexturePtr = 0;
        WebGPU.iterateExtensions(entryPtr, {
          14: ptr => {
            externalTexturePtr = HEAPU32[(((ptr) + (8)) >>> 2) >>> 0];
          }
        });
        var resource;
        if (bufferPtr) {
          // Note the sentinel UINT64_MAX will be read as -1.
          var size = readI53FromI64((entryPtr) + (24));
          if (size == -1) size = undefined;
          resource = {
            'buffer': WebGPU.getJsObject(bufferPtr),
            'offset': readI53FromI64((entryPtr) + (16)),
            'size': size
          };
        } else {
          resource = WebGPU.getJsObject(
              samplerPtr || textureViewPtr || externalTexturePtr);
        }
        return {
          'binding': HEAPU32[(((entryPtr) + (4)) >>> 2) >>> 0],
          'resource': resource
        };
      }
      function makeEntries(count, entriesPtrs) {
        var entries = [];
        for (var i = 0; i < count; ++i) {
          entries.push(makeEntry(entriesPtrs + 40 * i));
        }
        return entries;
      }
      var desc = {
        'label': WebGPU.makeStringFromOptionalStringView(descriptor + 4),
        'layout':
            WebGPU.getJsObject(HEAPU32[(((descriptor) + (12)) >>> 2) >>> 0]),
        'entries': makeEntries(
            HEAPU32[(((descriptor) + (16)) >>> 2) >>> 0],
            HEAPU32[(((descriptor) + (20)) >>> 2) >>> 0])
      };
      var device = WebGPU.getJsObject(devicePtr);
      var ptr = _emwgpuCreateBindGroup(0);
      WebGPU.Internals.jsObjectInsert(ptr, device.createBindGroup(desc));
      return ptr;
    };

    function _wgpuDeviceCreateBindGroupLayout(devicePtr, descriptor) {
      devicePtr >>>= 0;
      descriptor >>>= 0;
      function makeBufferEntry(substructPtr) {
        var typeInt = HEAPU32[(((substructPtr) + (4)) >>> 2) >>> 0];
        if (!typeInt) return undefined;
        return {
          'type': WebGPU.BufferBindingType[typeInt],
          'hasDynamicOffset': !!(HEAPU32[(((substructPtr) + (8)) >>> 2) >>> 0]),
          'minBindingSize': readI53FromI64((substructPtr) + (16))
        };
      }
      function makeSamplerEntry(substructPtr) {
        var typeInt = HEAPU32[(((substructPtr) + (4)) >>> 2) >>> 0];
        if (!typeInt) return undefined;
        return {'type': WebGPU.SamplerBindingType[typeInt]};
      }
      function makeTextureEntry(substructPtr) {
        var sampleTypeInt = HEAPU32[(((substructPtr) + (4)) >>> 2) >>> 0];
        if (!sampleTypeInt) return undefined;
        return {
          'sampleType': WebGPU.TextureSampleType[sampleTypeInt],
          'viewDimension': WebGPU.TextureViewDimension
                               [HEAP32[(((substructPtr) + (8)) >>> 2) >>> 0]],
          'multisampled': !!(HEAPU32[(((substructPtr) + (12)) >>> 2) >>> 0])
        };
      }
      function makeStorageTextureEntry(substructPtr) {
        var accessInt = HEAPU32[(((substructPtr) + (4)) >>> 2) >>> 0];
        if (!accessInt) return undefined;
        return {
          'access': WebGPU.StorageTextureAccess[accessInt],
          'format':
              WebGPU
                  .TextureFormat[HEAP32[(((substructPtr) + (8)) >>> 2) >>> 0]],
          'viewDimension': WebGPU.TextureViewDimension
                               [HEAP32[(((substructPtr) + (12)) >>> 2) >>> 0]]
        };
      }
      function makeEntry(entryPtr) {
        var entry = {
          'binding': HEAPU32[(((entryPtr) + (4)) >>> 2) >>> 0],
          'visibility': HEAPU32[(((entryPtr) + (8)) >>> 2) >>> 0],
          'buffer': makeBufferEntry(entryPtr + 24),
          'sampler': makeSamplerEntry(entryPtr + 48),
          'texture': makeTextureEntry(entryPtr + 56),
          'storageTexture': makeStorageTextureEntry(entryPtr + 72)
        };
        WebGPU.iterateExtensions(entryPtr, {
          13: ptr => {
            entry['externalTexture'] = {};
          }
        });
        return entry;
      }
      function makeEntries(count, entriesPtrs) {
        var entries = [];
        for (var i = 0; i < count; ++i) {
          entries.push(makeEntry(entriesPtrs + 88 * i));
        }
        return entries;
      }
      var desc = {
        'label': WebGPU.makeStringFromOptionalStringView(descriptor + 4),
        'entries': makeEntries(
            HEAPU32[(((descriptor) + (12)) >>> 2) >>> 0],
            HEAPU32[(((descriptor) + (16)) >>> 2) >>> 0])
      };
      var device = WebGPU.getJsObject(devicePtr);
      var ptr = _emwgpuCreateBindGroupLayout(0);
      WebGPU.Internals.jsObjectInsert(ptr, device.createBindGroupLayout(desc));
      return ptr;
    }

    function _wgpuDeviceCreateCommandEncoder(devicePtr, descriptor) {
      devicePtr >>>= 0;
      descriptor >>>= 0;
      var desc;
      if (descriptor) {
        desc = {
          'label': WebGPU.makeStringFromOptionalStringView(descriptor + 4)
        };
      }
      var device = WebGPU.getJsObject(devicePtr);
      var ptr = _emwgpuCreateCommandEncoder(0);
      WebGPU.Internals.jsObjectInsert(ptr, device.createCommandEncoder(desc));
      return ptr;
    }

    function _wgpuDeviceCreateComputePipeline(devicePtr, descriptor) {
      devicePtr >>>= 0;
      descriptor >>>= 0;
      var desc = WebGPU.makeComputePipelineDesc(descriptor);
      var device = WebGPU.getJsObject(devicePtr);
      var ptr = _emwgpuCreateComputePipeline(0);
      WebGPU.Internals.jsObjectInsert(ptr, device.createComputePipeline(desc));
      return ptr;
    }

    function _wgpuDeviceCreatePipelineLayout(devicePtr, descriptor) {
      devicePtr >>>= 0;
      descriptor >>>= 0;
      var bglCount = HEAPU32[(((descriptor) + (12)) >>> 2) >>> 0];
      var bglPtr = HEAPU32[(((descriptor) + (16)) >>> 2) >>> 0];
      var bgls = [];
      for (var i = 0; i < bglCount; ++i) {
        bgls.push(
            WebGPU.getJsObject(HEAPU32[(((bglPtr) + (4 * i)) >>> 2) >>> 0]));
      }
      var desc = {
        'label': WebGPU.makeStringFromOptionalStringView(descriptor + 4),
        'bindGroupLayouts': bgls,
        'immediateSize': HEAPU32[(((descriptor) + (20)) >>> 2) >>> 0]
      };
      var device = WebGPU.getJsObject(devicePtr);
      var ptr = _emwgpuCreatePipelineLayout(0);
      WebGPU.Internals.jsObjectInsert(ptr, device.createPipelineLayout(desc));
      return ptr;
    }

    function _wgpuDeviceCreateQuerySet(devicePtr, descriptor) {
      devicePtr >>>= 0;
      descriptor >>>= 0;
      var desc = {
        'type': WebGPU.QueryType[HEAP32[(((descriptor) + (12)) >>> 2) >>> 0]],
        'count': HEAPU32[(((descriptor) + (16)) >>> 2) >>> 0]
      };
      var device = WebGPU.getJsObject(devicePtr);
      var ptr = _emwgpuCreateQuerySet(0);
      WebGPU.Internals.jsObjectInsert(ptr, device.createQuerySet(desc));
      return ptr;
    }

    function _wgpuDeviceCreateTexture(devicePtr, descriptor) {
      devicePtr >>>= 0;
      descriptor >>>= 0;
      var nextInChainPtr = HEAPU32[((descriptor) >>> 2) >>> 0];
      var textureBindingViewDimension;
      if (nextInChainPtr !== 0) {
        var sType = HEAP32[(((nextInChainPtr) + (4)) >>> 2) >>> 0];
        var textureBindingViewDimensionDescriptor = nextInChainPtr;
        textureBindingViewDimension =
            WebGPU
                .TextureViewDimension[HEAP32
                                          [(((textureBindingViewDimensionDescriptor) +
                                             (8)) >>>
                                            2) >>>
                                           0]];
      }
      var desc = {
        'label': WebGPU.makeStringFromOptionalStringView(descriptor + 4),
        'size': WebGPU.makeExtent3D(descriptor + 28),
        'mipLevelCount': HEAPU32[(((descriptor) + (44)) >>> 2) >>> 0],
        'sampleCount': HEAPU32[(((descriptor) + (48)) >>> 2) >>> 0],
        'dimension':
            WebGPU
                .TextureDimension[HEAP32[(((descriptor) + (24)) >>> 2) >>> 0]],
        'format':
            WebGPU.TextureFormat[HEAP32[(((descriptor) + (40)) >>> 2) >>> 0]],
        'usage': HEAPU32[(((descriptor) + (16)) >>> 2) >>> 0],
        'textureBindingViewDimension': textureBindingViewDimension
      };
      var viewFormatCount = HEAPU32[(((descriptor) + (52)) >>> 2) >>> 0];
      if (viewFormatCount) {
        var viewFormatsPtr = HEAPU32[(((descriptor) + (56)) >>> 2) >>> 0];
        // viewFormatsPtr pointer to an array of TextureFormat which is an enum
        // of size uint32_t
        desc['viewFormats'] = Array.from(
            HEAP32.subarray(
                (((viewFormatsPtr) >>> 2)) >>> 0,
                ((viewFormatsPtr + viewFormatCount * 4) >>> 2) >>> 0),
            format => WebGPU.TextureFormat[format]);
      }
      var device = WebGPU.getJsObject(devicePtr);
      var ptr = _emwgpuCreateTexture(0);
      WebGPU.Internals.jsObjectInsert(ptr, device.createTexture(desc));
      return ptr;
    }

    function _wgpuDeviceGetAdapterInfo(devicePtr, adapterInfo) {
      devicePtr >>>= 0;
      adapterInfo >>>= 0;
      var device = WebGPU.getJsObject(devicePtr);
      WebGPU.fillAdapterInfoStruct(device.adapterInfo, adapterInfo);
      return 1;
    }

    function _wgpuDeviceGetLimits(devicePtr, limitsOutPtr) {
      devicePtr >>>= 0;
      limitsOutPtr >>>= 0;
      var device = WebGPU.getJsObject(devicePtr);
      WebGPU.fillLimitStruct(device.limits, limitsOutPtr);
      return 1;
    }

    function _wgpuDeviceHasFeature(devicePtr, featureEnumValue) {
      devicePtr >>>= 0;
      var device = WebGPU.getJsObject(devicePtr);
      return device.features.has(WebGPU.FeatureName[featureEnumValue]);
    }

    var _wgpuQueueSubmit = function(queuePtr, commandCount, commands) {
      queuePtr >>>= 0;
      commandCount >>>= 0;
      commands >>>= 0;
      var queue = WebGPU.getJsObject(queuePtr);
      var cmds = Array.from(
          HEAP32.subarray(
              (((commands) >>> 2)) >>> 0,
              ((commands + commandCount * 4) >>> 2) >>> 0),
          id => WebGPU.getJsObject(id));
      queue.submit(cmds);
    };

    function _wgpuQueueWriteBuffer(
        queuePtr, bufferPtr, bufferOffset_low, bufferOffset_high, data, size) {
      queuePtr >>>= 0;
      bufferPtr >>>= 0;
      var bufferOffset =
          convertI32PairToI53Checked(bufferOffset_low, bufferOffset_high);
      data >>>= 0;
      size >>>= 0;
      var queue = WebGPU.getJsObject(queuePtr);
      var buffer = WebGPU.getJsObject(bufferPtr);
      // There is a size limitation for ArrayBufferView. Work around by passing
      // in a subarray instead of the whole heap. crbug.com/1201109
      var subarray = HEAPU8.subarray(data >>> 0, data + size >>> 0);
      queue.writeBuffer(buffer, bufferOffset, subarray, 0, size);
    }

    function _wgpuQueueWriteTexture(
        queuePtr, destinationPtr, data, dataSize, dataLayoutPtr, writeSizePtr) {
      queuePtr >>>= 0;
      destinationPtr >>>= 0;
      data >>>= 0;
      dataSize >>>= 0;
      dataLayoutPtr >>>= 0;
      writeSizePtr >>>= 0;
      var queue = WebGPU.getJsObject(queuePtr);
      var destination = WebGPU.makeTexelCopyTextureInfo(destinationPtr);
      var dataLayout = WebGPU.makeTexelCopyBufferLayout(dataLayoutPtr);
      var writeSize = WebGPU.makeExtent3D(writeSizePtr);
      // This subarray isn't strictly necessary, but helps work around an issue
      // where Chromium makes a copy of the entire heap. crbug.com/1134457
      var subarray = HEAPU8.subarray(data >>> 0, data + dataSize >>> 0);
      queue.writeTexture(destination, subarray, dataLayout, writeSize);
    }

    function _wgpuTextureCreateView(texturePtr, descriptor) {
      texturePtr >>>= 0;
      descriptor >>>= 0;
      var desc;
      if (descriptor) {
        var swizzle;
        var nextInChainPtr = HEAPU32[((descriptor) >>> 2) >>> 0];
        if (nextInChainPtr !== 0) {
          var sType = HEAP32[(((nextInChainPtr) + (4)) >>> 2) >>> 0];
          var swizzleDescriptor = nextInChainPtr;
          var swizzlePtr = swizzleDescriptor + 8;
          var r = WebGPU.ComponentSwizzle[HEAP32[((swizzlePtr) >>> 2) >>> 0]] ||
              'r';
          var g =
              WebGPU
                  .ComponentSwizzle[HEAP32
                                        [(((swizzlePtr) + (4)) >>> 2) >>> 0]] ||
              'g';
          var b =
              WebGPU
                  .ComponentSwizzle[HEAP32
                                        [(((swizzlePtr) + (8)) >>> 2) >>> 0]] ||
              'b';
          var a = WebGPU.ComponentSwizzle[HEAP32
                                              [(((swizzlePtr) + (12)) >>> 2) >>>
                                               0]] ||
              'a';
          swizzle = `${r}${g}${b}${a}`;
        }
        var mipLevelCount = HEAPU32[(((descriptor) + (24)) >>> 2) >>> 0];
        var arrayLayerCount = HEAPU32[(((descriptor) + (32)) >>> 2) >>> 0];
        desc = {
          'label': WebGPU.makeStringFromOptionalStringView(descriptor + 4),
          'format':
              WebGPU.TextureFormat[HEAP32[(((descriptor) + (12)) >>> 2) >>> 0]],
          'dimension': WebGPU.TextureViewDimension
                           [HEAP32[(((descriptor) + (16)) >>> 2) >>> 0]],
          'baseMipLevel': HEAPU32[(((descriptor) + (20)) >>> 2) >>> 0],
          'mipLevelCount': mipLevelCount === 4294967295 ? undefined :
                                                          mipLevelCount,
          'baseArrayLayer': HEAPU32[(((descriptor) + (28)) >>> 2) >>> 0],
          'arrayLayerCount': arrayLayerCount === 4294967295 ? undefined :
                                                              arrayLayerCount,
          'aspect':
              WebGPU.TextureAspect[HEAP32[(((descriptor) + (36)) >>> 2) >>> 0]],
          'usage': HEAPU32[(((descriptor) + (40)) >>> 2) >>> 0],
          'swizzle': swizzle
        };
      }
      var texture = WebGPU.getJsObject(texturePtr);
      var ptr = _emwgpuCreateTextureView(0);
      WebGPU.Internals.jsObjectInsert(ptr, texture.createView(desc));
      return ptr;
    }

    function _wgpuTextureDestroy(texturePtr) {
      texturePtr >>>= 0;
      WebGPU.getJsObject(texturePtr).destroy();
    }

    function _wgpuTextureGetDepthOrArrayLayers(texturePtr) {
      texturePtr >>>= 0;
      var texture = WebGPU.getJsObject(texturePtr);
      return texture.depthOrArrayLayers;
    }

    function _wgpuTextureGetFormat(texturePtr) {
      texturePtr >>>= 0;
      var texture = WebGPU.getJsObject(texturePtr);
      // Should return the enum integer instead of string.
      return WebGPU.TextureFormat.indexOf(texture.format);
    }

    function _wgpuTextureGetHeight(texturePtr) {
      texturePtr >>>= 0;
      var texture = WebGPU.getJsObject(texturePtr);
      return texture.height;
    }

    function _wgpuTextureGetWidth(texturePtr) {
      texturePtr >>>= 0;
      var texture = WebGPU.getJsObject(texturePtr);
      return texture.width;
    }

    var getCFunc = ident => {
      var func = Module['_' + ident];
      // closure exported function
      return func;
    };

    var writeArrayToMemory = (array, buffer) => {
      HEAP8.set(array, buffer >>> 0);
    };

    /**
     * @param {string|null=} returnType
     * @param {Array=} argTypes
     * @param {Array=} args
     * @param {Object=} opts
     */
    var ccall = (ident, returnType, argTypes, args, opts) => {
      // For fast lookup of conversion functions
      var toC = {
        'string': str => {
          var ret = 0;
          if (str !== null && str !== undefined && str !== 0) {
            // null string
            ret = stringToUTF8OnStack(str);
          }
          return ret;
        },
        'array': arr => {
          var ret = stackAlloc(arr.length);
          writeArrayToMemory(arr, ret);
          return ret;
        }
      };
      function convertReturnValue(ret) {
        if (returnType === 'string') {
          return UTF8ToString(ret);
        }
        if (returnType === 'pointer') return ret >>> 0;
        if (returnType === 'boolean') return Boolean(ret);
        return ret;
      }
      var func = getCFunc(ident);
      var cArgs = [];
      var stack = 0;
      if (args) {
        for (var i = 0; i < args.length; i++) {
          var converter = toC[argTypes[i]];
          if (converter) {
            if (stack === 0) stack = stackSave();
            cArgs[i] = converter(args[i]);
          } else {
            cArgs[i] = args[i];
          }
        }
      }
      // Data for a previous async operation that was in flight before us.
      var previousAsync = Asyncify.currData;
      var ret = func(...cArgs);
      function onDone(ret) {
        runtimeKeepalivePop();
        if (stack !== 0) stackRestore(stack);
        return convertReturnValue(ret);
      }
      var asyncMode = opts?.async;
      // Keep the runtime alive through all calls. Note that this call might not
      // be async, but for simplicity we push and pop in all calls.
      runtimeKeepalivePush();
      if (Asyncify.currData != previousAsync) {
        // This is a new async operation. The wasm is paused and has unwound its
        // stack. We need to return a Promise that resolves the return value
        // once the stack is rewound and execution finishes.
        return Asyncify.whenDone().then(onDone);
      }
      ret = onDone(ret);
      // If this is an async ccall, ensure we return a promise
      if (asyncMode) return Promise.resolve(ret);
      return ret;
    };

    var FS_createPath = (...args) => FS.createPath(...args);

    var FS_unlink = (...args) => FS.unlink(...args);

    var FS_createLazyFile = (...args) => FS.createLazyFile(...args);

    var FS_createDevice = (...args) => FS.createDevice(...args);

    FS.createPreloadedFile = FS_createPreloadedFile;

    FS.preloadFile = FS_preloadFile;

    FS.staticInit();

    // Signal GL rendering layer that processing of a new frame is about to
    // start. This helps it optimize VBO double-buffering and reduce GPU stalls.
    registerPreMainLoop(() => GL.newRenderingFrameStarted());

    // End JS library code
    // include: postlibrary.js
    // This file is included after the automatically-generated JS library code
    // but before the wasm module is created.
    {
      // Begin ATMODULES hooks
      if (Module['noExitRuntime']) noExitRuntime = Module['noExitRuntime'];
      if (Module['print']) out = Module['print'];
      if (Module['printErr']) err = Module['printErr'];
      // End ATMODULES hooks
      if (Module['arguments']) programArgs = Module['arguments'];
      if (Module['thisProgram']) thisProgram = Module['thisProgram'];
      var preInit = Module['preInit'];
      if (preInit) {
        if (typeof preInit == 'function')
          Module['preInit'] = preInit = [preInit];
        // Written as a loop so that preInit functions that themselves add more
        // preInit functions.  Is this actually needed?
        while (preInit.length > 0) {
          preInit.shift()();
        }
      }
    }

    // Begin runtime exports
    Module['addRunDependency'] = addRunDependency;

    Module['removeRunDependency'] = removeRunDependency;

    Module['ccall'] = ccall;

    Module['stringToNewUTF8'] = stringToNewUTF8;

    Module['FS_preloadFile'] = FS_preloadFile;

    Module['FS_unlink'] = FS_unlink;

    Module['FS_createPath'] = FS_createPath;

    Module['FS_createDevice'] = FS_createDevice;

    Module['FS_createDataFile'] = FS_createDataFile;

    Module['FS_createLazyFile'] = FS_createLazyFile;

    // End runtime exports
    // Begin JS library exports
    // End JS library exports
    // end include: postlibrary.js
    var ASM_CONSTS = {
      2928872: () => !!Module['preinitializedWebGPUDevice'],
      2928923: () => {
        const retVal = Module.LLM_CANCEL_FLAG;
        Module.LLM_CANCEL_FLAG = undefined;
        return retVal || 0;
      },
      2929022: $0 => {
        const device = WebGPU.getJsObject($0);
        return device.features.has('subgroups');
      },
      2929106: () => !!Module['preinitializedWebGPUDevice'],
      2929157: () => {
        specialHTMLTargets['#canvas'] = Module.canvas;
      }
    };

    function __asyncjs__InvokeReadDataFn(fn, offset, size, mode) {
      return Asyncify.handleAsync(
          async () => await Emval.toValue(fn)(offset, size, mode));
    }

    function JsProgressListener(text, done) {
      Module._userProgressListener(UTF8ToString(text), done);
    }

    function DefaultErrorReporter(message) {
      throw new Error(UTF8ToString(message));
    }

    function ThrowError(val_handle) {
      const error = Emval.toValue(val_handle);
      throw error;
    }

    function JsWrapLoggingHelpers() {
      if (!Module['_decodeBase64']) {
        Module['_decodeBase64'] = function(encoded_key_ptr) {
          const encodedStr = UTF8ToString(encoded_key_ptr);
          return atob(encodedStr);
        };
      }
    }

    function JsOnEmptyPacketListener(output_stream_name, timestamp) {
      Module._wrapEmptyPacketListenerOutput(output_stream_name, timestamp);
    }

    function JsOnVectorFinishedListener(output_stream_name, timestamp) {
      Module._wrapSimpleListenerOutput(
          output_stream_name, undefined, true, timestamp);
    }

    function JsOnSimpleListenerBool(output_stream_name, out_data, timestamp) {
      Module._wrapSimpleListenerOutput(output_stream_name, out_data, timestamp);
    }

    function JsOnVectorListenerBool(output_stream_name, out_data, timestamp) {
      Module._wrapSimpleListenerOutput(
          output_stream_name, out_data, false, timestamp);
    }

    function JsOnSimpleListenerInt(output_stream_name, out_data, timestamp) {
      Module._wrapSimpleListenerOutput(output_stream_name, out_data, timestamp);
    }

    function JsOnVectorListenerInt(output_stream_name, out_data, timestamp) {
      Module._wrapSimpleListenerOutput(
          output_stream_name, out_data, false, timestamp);
    }

    function JsOnSimpleListenerUint(output_stream_name, out_data, timestamp) {
      Module._wrapSimpleListenerOutput(output_stream_name, out_data, timestamp);
    }

    function JsOnVectorListenerUint(output_stream_name, out_data, timestamp) {
      Module._wrapSimpleListenerOutput(
          output_stream_name, out_data, false, timestamp);
    }

    function JsOnSimpleListenerDouble(output_stream_name, out_data, timestamp) {
      Module._wrapSimpleListenerOutput(output_stream_name, out_data, timestamp);
    }

    function JsOnVectorListenerDouble(output_stream_name, out_data, timestamp) {
      Module._wrapSimpleListenerOutput(
          output_stream_name, out_data, false, timestamp);
    }

    function JsOnSimpleListenerFloat(output_stream_name, out_data, timestamp) {
      Module._wrapSimpleListenerOutput(output_stream_name, out_data, timestamp);
    }

    function JsOnVectorListenerFloat(output_stream_name, out_data, timestamp) {
      Module._wrapSimpleListenerOutput(
          output_stream_name, out_data, false, timestamp);
    }

    function JsOnSimpleListenerString(output_stream_name, out_data, timestamp) {
      Module._wrapSimpleListenerOutput(
          output_stream_name, UTF8ToString(out_data), timestamp);
    }

    function JsOnVectorListenerString(output_stream_name, out_data, timestamp) {
      Module._wrapSimpleListenerOutput(
          output_stream_name, UTF8ToString(out_data), false, timestamp);
    }

    function JsOnVectorListenerProto(
        output_stream_name, proto_ptr, proto_size, make_deep_copy, timestamp) {
      const newProtoArray = make_deep_copy ?
          Module.HEAPU8.slice(proto_ptr, proto_ptr + proto_size) :
          new Uint8Array(Module.HEAPU8.buffer, proto_ptr, proto_size);
      Module._wrapSimpleListenerOutput(
          output_stream_name, newProtoArray, false, timestamp);
    }

    function JsWrapSimpleListeners() {
      if (!Module._wrapSimpleListenerOutput) {
        Module._wrapSimpleListenerOutput = (outputStreamName, ...args) => {
          if (Module.simpleListeners) {
            const streamName = UTF8ToString(outputStreamName);
            if (Module.simpleListeners[streamName]) {
              Module.simpleListeners[streamName](...args);
            }
          }
        };
      }
      if (!Module._wrapEmptyPacketListenerOutput) {
        Module._wrapEmptyPacketListenerOutput =
            (outputStreamName, timestamp) => {
              if (Module.emptyPacketListeners) {
                const streamName = UTF8ToString(outputStreamName);
                if (Module.emptyPacketListeners[streamName]) {
                  Module.emptyPacketListeners[streamName](timestamp);
                }
              }
            };
      }
    }

    function JsOnSimpleListenerBinaryArray(
        output_stream_name, binary_ptr, binary_size, make_deep_copy,
        timestamp) {
      const newProtoArray = make_deep_copy ?
          Module.HEAPU8.slice(binary_ptr, binary_ptr + binary_size) :
          new Uint8Array(Module.HEAPU8.buffer, binary_ptr, binary_size);
      Module._wrapSimpleListenerOutput(
          output_stream_name, newProtoArray, timestamp);
    }

    function __asyncjs__CallReadDataFn(fn, offset, size, mode) {
      return Asyncify.handleAsync(
          async () => await Emval.toValue(fn)(offset, size, mode));
    }

    function __asyncjs__mediapipe_map_buffer_jspi(buffer_handle, data) {
      return Asyncify.handleAsync(async () => {
        const buffer = WebGPU.getJsObject(buffer_handle);
        if ('mapSync' in buffer) {
          buffer.mapSync(GPUMapMode.READ);
        } else {
          await buffer.mapAsync(GPUMapMode.READ);
        }
        const mapped = buffer.getMappedRange();
        HEAPU8.set(new Uint8Array(mapped), data >>> 0);
        buffer.unmap();
      });
    }

    function JsGetDeviceMinSubgroupSize(deviceId) {
      const device = WebGPU.getJsObject(deviceId);
      return device.adapterInfo.subgroupMinSize ||
          device.limits.minSubgroupSize;
    }

    function JsGetDeviceMaxSubgroupSize(deviceId) {
      const device = WebGPU.getJsObject(deviceId);
      return device.adapterInfo.subgroupMaxSize ||
          device.limits.maxSubgroupSize;
    }

    function __asyncjs__ReadBufferDataJs(buffer_handle, data_ptr) {
      return Asyncify.handleAsync(async () => {
        const gpuReadBuffer = WebGPU.getJsObject(buffer_handle);
        await gpuReadBuffer.mapAsync(GPUMapMode.READ);
        const arrayBuffer = gpuReadBuffer.getMappedRange();
        const u8view = new Uint8Array(arrayBuffer);
        HEAPU8.set(u8view, data_ptr >>> 0 >>> 0);
        gpuReadBuffer.unmap();
      });
    }

    function GetAdapterArchitecture() {
      const device = Module['preinitializedWebGPUDevice'];
      const architecture =
          device.adapterInfo ? device.adapterInfo.architecture : 'Unknown';
      return stringToNewUTF8(architecture);
    }

    function GetAdapterDescription() {
      const device = Module['preinitializedWebGPUDevice'];
      const description =
          device.adapterInfo ? device.adapterInfo.description : 'Unknown';
      return stringToNewUTF8(description);
    }

    function GetAdapterDeviceName() {
      const device = Module['preinitializedWebGPUDevice'];
      const deviceName =
          device.adapterInfo ? device.adapterInfo.device : 'Unknown';
      return stringToNewUTF8(deviceName);
    }

    function GetAdapterVendor() {
      const device = Module['preinitializedWebGPUDevice'];
      const vendor = device.adapterInfo ? device.adapterInfo.vendor : 'Unknown';
      return stringToNewUTF8(vendor);
    }

    function __asyncjs__CallStreamWeightsOnWeb(
        tfl_ids, wgpu_buffers, offsets, lengths, count) {
      return Asyncify.handleAsync(async () => {
        const callback = Module.getStreamWeightsCallback();
        if (typeof callback !== 'function') {
          console.error(
              'Stream weights callback is not registered or is not a function');
          return 1;
        }
        const view = new DataView(Module.HEAPU8.buffer);
        const tflIdsArray = new Int32Array(count);
        const wgpuBuffersArray = new Uint32Array(count);
        const offsetsArray = new Float64Array(count);
        const lengthsArray = new Float64Array(count);
        for (let i = 0; i < count; i++) {
          tflIdsArray[i] = view.getInt32(tfl_ids + i * 4, true);
          wgpuBuffersArray[i] = view.getUint32(wgpu_buffers + i * 4, true);
          offsetsArray[i] = view.getFloat64(offsets + i * 8, true);
          lengthsArray[i] = view.getFloat64(lengths + i * 8, true);
        }
        try {
          await callback(
              tflIdsArray, wgpuBuffersArray, offsetsArray, lengthsArray);
        } catch (e) {
          console.error('Error in streamWeightsOnWeb:', e);
          return 1;
        }
        return 0;
      });
    }

    function hardware_concurrency() {
      var concurrency = 1;
      try {
        concurrency = self.navigator.hardwareConcurrency;
      } catch (e) {
      }
      return concurrency;
    }

    function UseBottomLeftGpuOrigin() {
      return (Module && Module.gpuOriginForWebTexturesIsBottomLeft);
    }

    function custom_emscripten_dbgn(str, len) {
      if (typeof (dbg) !== 'undefined') {
        dbg(UTF8ToString(str, len));
      } else {
        if (typeof (custom_dbg) === 'undefined') {
          function custom_dbg(text) {
            console.warn.apply(console, arguments);
          }
        }
        custom_dbg(UTF8ToString(str, len));
      }
    }

    function JsWrapErrorListenerInternal(code, message) {
      if (Module.errorListener) {
        const stringMessage = UTF8ToString(message);
        Module.errorListener(code, stringMessage);
      }
    }

    // Imports from the Wasm binary.
    var _free, _CreateLlmInferenceEngineConverted, _DeleteLlmInferenceEngine,
        _MakeSessionForPredict, _AddTextQueryChunk, _AddImageQueryChunk,
        _AddAudioQueryChunk, _malloc, _PredictSession, _FreeSession,
        _GetLiteRtModelOffset, _GetSizeInTokens, _wgpuDeviceAddRef,
        _mediapipeLoggerGetEncodedApiKey, _registerModelResourcesGraphService,
        _bindTextureToStream, _addBoundTextureToStream, _addDoubleToInputStream,
        _addFloatToInputStream, _addBoolToInputStream, _addIntToInputStream,
        _addUintToInputStream, _addStringToInputStream,
        _addRawDataSpanToInputStream, _allocateBoolVector, _allocateFloatVector,
        _allocateDoubleVector, _allocateIntVector, _allocateUintVector,
        _allocateStringVector, _addBoolVectorEntry, _addFloatVectorEntry,
        _addDoubleVectorEntry, _addIntVectorEntry, _addUintVectorEntry,
        _addStringVectorEntry, _addBoolVectorToInputStream,
        _addFloatVectorToInputStream, _addDoubleVectorToInputStream,
        _addIntVectorToInputStream, _addUintVectorToInputStream,
        _addStringVectorToInputStream, _addFlatHashMapToInputStream,
        _addProtoToInputStream, _addEmptyPacketToInputStream,
        _addBoolToInputSidePacket, _addDoubleToInputSidePacket,
        _addFloatToInputSidePacket, _addIntToInputSidePacket,
        _addUintToInputSidePacket, _addStringToInputSidePacket,
        _addRawDataSpanToInputSidePacket, _addProtoToInputSidePacket,
        _addBoolVectorToInputSidePacket, _addDoubleVectorToInputSidePacket,
        _addFloatVectorToInputSidePacket, _addIntVectorToInputSidePacket,
        _addUintVectorToInputSidePacket, _addStringVectorToInputSidePacket,
        _attachBoolListener, _attachBoolVectorListener, _attachDoubleListener,
        _attachDoubleVectorListener, _attachFloatListener,
        _attachFloatVectorListener, _attachIntListener,
        _attachIntVectorListener, _attachUintListener,
        _attachUintVectorListener, _attachStringListener,
        _attachStringVectorListener, _attachProtoListener,
        _attachProtoVectorListener, _getGraphConfig, ___getTypeName,
        _emwgpuCreateBindGroup, _emwgpuCreateBindGroupLayout,
        _emwgpuCreateCommandBuffer, _emwgpuCreateCommandEncoder,
        _emwgpuCreateComputePassEncoder, _emwgpuCreateComputePipeline,
        _emwgpuCreateExternalTexture, _emwgpuCreatePipelineLayout,
        _emwgpuCreateQuerySet, _emwgpuCreateRenderBundle,
        _emwgpuCreateRenderBundleEncoder, _emwgpuCreateRenderPassEncoder,
        _emwgpuCreateRenderPipeline, _emwgpuCreateSampler, _emwgpuCreateSurface,
        _emwgpuCreateTexture, _emwgpuCreateTextureView, _emwgpuCreateAdapter,
        _emwgpuImportBuffer, _emwgpuCreateDevice, _emwgpuCreateQueue,
        _emwgpuCreateShaderModule, _emwgpuOnCreateComputePipelineCompleted,
        _emwgpuOnWorkDoneCompleted, _clearSubgraphs, _pushBinarySubgraph,
        _pushTextSubgraph, _changeBinaryGraph, _changeTextGraph, _processGl,
        _process, _bindTextureToCanvas, _requestShaderRefreshOnGraphChange,
        _waitUntilIdle, _closeGraph, _setAutoRenderToScreen,
        _emscripten_builtin_memalign, _memalign, __emscripten_tempret_set,
        __emscripten_stack_restore, __emscripten_stack_alloc,
        _emscripten_stack_get_current, dynCall_vii, dynCall_viiiiiii,
        dynCall_viiiiii, dynCall_v, dynCall_ii, dynCall_vi, dynCall_iiiii,
        dynCall_iii, dynCall_iiii, dynCall_viiii, dynCall_viiiii, dynCall_viii,
        dynCall_iiiijij, dynCall_i, dynCall_ji, dynCall_jii, dynCall_ff,
        dynCall_did, dynCall_iiiiiii, dynCall_iiiiii, dynCall_iiiiiiiiiii,
        dynCall_iiiiiiiiiiii, dynCall_iiiiiiiiiiiii, dynCall_viijj,
        dynCall_iiiiiiii, dynCall_viji, dynCall_vijiii, dynCall_iiiiiiiii,
        dynCall_viiiiiiii, dynCall_iiiif, dynCall_viiiiiiiiiiiiiii,
        dynCall_viff, dynCall_viiffii, dynCall_viiffi, dynCall_viiiiiiiiii,
        dynCall_fi, dynCall_viiffff, dynCall_viiiiij, dynCall_vij,
        dynCall_viiiiiffii, dynCall_vif, dynCall_ddd, dynCall_dddd, dynCall_idd,
        dynCall_di, dynCall_jiiiijiiiii, dynCall_viiij, dynCall_viid,
        dynCall_vidd, dynCall_vifd, dynCall_viiiiiiiiiii, dynCall_iiif,
        dynCall_jiii, dynCall_iiiiiiiiii, dynCall_iij, dynCall_iiiijj,
        dynCall_viiijjj, dynCall_fff, dynCall_viifii, dynCall_fii,
        dynCall_vfiii, dynCall_jjj, dynCall_fiii, dynCall_viffi,
        dynCall_viiiiiiiii, dynCall_viiiffii, dynCall_iiiiiffi,
        dynCall_viiiiiff, dynCall_viiiiiiiiiiiii, dynCall_viiiiiiiiiiii,
        dynCall_viiiiiiiiiiiiiiiii, dynCall_iifff, dynCall_iiff,
        dynCall_iiifiii, dynCall_iiffi, dynCall_iiffii, dynCall_viiiiiiiiiiiiii,
        dynCall_iif, dynCall_viijii, dynCall_iiiji, dynCall_viiif,
        dynCall_viiifiii, dynCall_fiif, dynCall_iiiifi, dynCall_iiiifiii,
        dynCall_iiifi, dynCall_viiiif, dynCall_viiiifii, dynCall_dii,
        dynCall_vifi, dynCall_vidi, dynCall_vijjj, dynCall_vj, dynCall_viij,
        dynCall_jiji, dynCall_iidiiiii, dynCall_iiiiij, dynCall_iiiiid,
        dynCall_iiiiijj, dynCall_iiiiiijj, _asyncify_start_unwind,
        _asyncify_stop_unwind, _asyncify_start_rewind, _asyncify_stop_rewind,
        memory, _kVersionStampBuildChangelistStr, _kVersionStampCitcSnapshotStr,
        _kVersionStampCitcWorkspaceIdStr, _kVersionStampSourceUriStr,
        _kVersionStampBuildClientStr, _kVersionStampBuildClientMintStatusStr,
        _kVersionStampBuildCompilerStr, _kVersionStampBuildDateTimePstStr,
        _kVersionStampBuildDepotPathStr, _kVersionStampBuildIdStr,
        _kVersionStampBuildInfoStr, _kVersionStampBuildLabelStr,
        _kVersionStampBuildTargetStr, _kVersionStampBuildTimestampStr,
        _kVersionStampBuildToolStr, _kVersionStampG3BuildTargetStr,
        _kVersionStampVerifiableStr, _kVersionStampBuildFdoTypeStr,
        _kVersionStampBuildBaselineChangelistStr, _kVersionStampBuildLtoTypeStr,
        _kVersionStampBuildPropellerTypeStr, _kVersionStampBuildPghoTypeStr,
        _kVersionStampBuildFdoProfileChangelistStr,
        _kVersionStampBuildMemprofProfileChangelistStr,
        _kVersionStampBuildUsernameStr, _kVersionStampBuildHostnameStr,
        _kVersionStampBuildDirectoryStr, _kVersionStampBuildChangelistInt,
        _kVersionStampCitcSnapshotInt, _kVersionStampBuildClientMintStatusInt,
        _kVersionStampBuildTimestampInt, _kVersionStampVerifiableInt,
        _kVersionStampBuildCoverageEnabledInt,
        _kVersionStampBuildBaselineChangelistInt,
        _kVersionStampPrecookedTimestampStr,
        _kVersionStampPrecookedClientInfoStr, __indirect_function_table,
        _kVersionStampBuildHasHardeningProtobuf, wasmMemory;

    function assignWasmExports(wasmExports) {
      _free = Module['_free'] = wasmExports['od'];
      _CreateLlmInferenceEngineConverted =
          Module['_CreateLlmInferenceEngineConverted'] = wasmExports['pd'];
      _DeleteLlmInferenceEngine = Module['_DeleteLlmInferenceEngine'] =
          wasmExports['qd'];
      _MakeSessionForPredict = Module['_MakeSessionForPredict'] =
          wasmExports['rd'];
      _AddTextQueryChunk = Module['_AddTextQueryChunk'] = wasmExports['sd'];
      _AddImageQueryChunk = Module['_AddImageQueryChunk'] = wasmExports['td'];
      _AddAudioQueryChunk = Module['_AddAudioQueryChunk'] = wasmExports['ud'];
      _malloc = Module['_malloc'] = wasmExports['vd'];
      _PredictSession = Module['_PredictSession'] = wasmExports['wd'];
      _FreeSession = Module['_FreeSession'] = wasmExports['xd'];
      _GetLiteRtModelOffset = Module['_GetLiteRtModelOffset'] =
          wasmExports['yd'];
      _GetSizeInTokens = Module['_GetSizeInTokens'] = wasmExports['zd'];
      _wgpuDeviceAddRef = wasmExports['Ad'];
      _mediapipeLoggerGetEncodedApiKey =
          Module['_mediapipeLoggerGetEncodedApiKey'] = wasmExports['Bd'];
      _registerModelResourcesGraphService =
          Module['_registerModelResourcesGraphService'] = wasmExports['Cd'];
      _bindTextureToStream = Module['_bindTextureToStream'] = wasmExports['Dd'];
      _addBoundTextureToStream = Module['_addBoundTextureToStream'] =
          wasmExports['Ed'];
      _addDoubleToInputStream = Module['_addDoubleToInputStream'] =
          wasmExports['Fd'];
      _addFloatToInputStream = Module['_addFloatToInputStream'] =
          wasmExports['Gd'];
      _addBoolToInputStream = Module['_addBoolToInputStream'] =
          wasmExports['Hd'];
      _addIntToInputStream = Module['_addIntToInputStream'] = wasmExports['Id'];
      _addUintToInputStream = Module['_addUintToInputStream'] =
          wasmExports['Jd'];
      _addStringToInputStream = Module['_addStringToInputStream'] =
          wasmExports['Kd'];
      _addRawDataSpanToInputStream = Module['_addRawDataSpanToInputStream'] =
          wasmExports['Ld'];
      _allocateBoolVector = Module['_allocateBoolVector'] = wasmExports['Md'];
      _allocateFloatVector = Module['_allocateFloatVector'] = wasmExports['Nd'];
      _allocateDoubleVector = Module['_allocateDoubleVector'] =
          wasmExports['Od'];
      _allocateIntVector = Module['_allocateIntVector'] = wasmExports['Pd'];
      _allocateUintVector = Module['_allocateUintVector'] = wasmExports['Qd'];
      _allocateStringVector = Module['_allocateStringVector'] =
          wasmExports['Rd'];
      _addBoolVectorEntry = Module['_addBoolVectorEntry'] = wasmExports['Sd'];
      _addFloatVectorEntry = Module['_addFloatVectorEntry'] = wasmExports['Td'];
      _addDoubleVectorEntry = Module['_addDoubleVectorEntry'] =
          wasmExports['Ud'];
      _addIntVectorEntry = Module['_addIntVectorEntry'] = wasmExports['Vd'];
      _addUintVectorEntry = Module['_addUintVectorEntry'] = wasmExports['Wd'];
      _addStringVectorEntry = Module['_addStringVectorEntry'] =
          wasmExports['Xd'];
      _addBoolVectorToInputStream = Module['_addBoolVectorToInputStream'] =
          wasmExports['Yd'];
      _addFloatVectorToInputStream = Module['_addFloatVectorToInputStream'] =
          wasmExports['Zd'];
      _addDoubleVectorToInputStream = Module['_addDoubleVectorToInputStream'] =
          wasmExports['_d'];
      _addIntVectorToInputStream = Module['_addIntVectorToInputStream'] =
          wasmExports['$d'];
      _addUintVectorToInputStream = Module['_addUintVectorToInputStream'] =
          wasmExports['ae'];
      _addStringVectorToInputStream = Module['_addStringVectorToInputStream'] =
          wasmExports['be'];
      _addFlatHashMapToInputStream = Module['_addFlatHashMapToInputStream'] =
          wasmExports['ce'];
      _addProtoToInputStream = Module['_addProtoToInputStream'] =
          wasmExports['de'];
      _addEmptyPacketToInputStream = Module['_addEmptyPacketToInputStream'] =
          wasmExports['ee'];
      _addBoolToInputSidePacket = Module['_addBoolToInputSidePacket'] =
          wasmExports['fe'];
      _addDoubleToInputSidePacket = Module['_addDoubleToInputSidePacket'] =
          wasmExports['ge'];
      _addFloatToInputSidePacket = Module['_addFloatToInputSidePacket'] =
          wasmExports['he'];
      _addIntToInputSidePacket = Module['_addIntToInputSidePacket'] =
          wasmExports['ie'];
      _addUintToInputSidePacket = Module['_addUintToInputSidePacket'] =
          wasmExports['je'];
      _addStringToInputSidePacket = Module['_addStringToInputSidePacket'] =
          wasmExports['ke'];
      _addRawDataSpanToInputSidePacket =
          Module['_addRawDataSpanToInputSidePacket'] = wasmExports['le'];
      _addProtoToInputSidePacket = Module['_addProtoToInputSidePacket'] =
          wasmExports['me'];
      _addBoolVectorToInputSidePacket =
          Module['_addBoolVectorToInputSidePacket'] = wasmExports['ne'];
      _addDoubleVectorToInputSidePacket =
          Module['_addDoubleVectorToInputSidePacket'] = wasmExports['oe'];
      _addFloatVectorToInputSidePacket =
          Module['_addFloatVectorToInputSidePacket'] = wasmExports['pe'];
      _addIntVectorToInputSidePacket =
          Module['_addIntVectorToInputSidePacket'] = wasmExports['qe'];
      _addUintVectorToInputSidePacket =
          Module['_addUintVectorToInputSidePacket'] = wasmExports['re'];
      _addStringVectorToInputSidePacket =
          Module['_addStringVectorToInputSidePacket'] = wasmExports['se'];
      _attachBoolListener = Module['_attachBoolListener'] = wasmExports['te'];
      _attachBoolVectorListener = Module['_attachBoolVectorListener'] =
          wasmExports['ue'];
      _attachDoubleListener = Module['_attachDoubleListener'] =
          wasmExports['ve'];
      _attachDoubleVectorListener = Module['_attachDoubleVectorListener'] =
          wasmExports['we'];
      _attachFloatListener = Module['_attachFloatListener'] = wasmExports['xe'];
      _attachFloatVectorListener = Module['_attachFloatVectorListener'] =
          wasmExports['ye'];
      _attachIntListener = Module['_attachIntListener'] = wasmExports['ze'];
      _attachIntVectorListener = Module['_attachIntVectorListener'] =
          wasmExports['Ae'];
      _attachUintListener = Module['_attachUintListener'] = wasmExports['Be'];
      _attachUintVectorListener = Module['_attachUintVectorListener'] =
          wasmExports['Ce'];
      _attachStringListener = Module['_attachStringListener'] =
          wasmExports['De'];
      _attachStringVectorListener = Module['_attachStringVectorListener'] =
          wasmExports['Ee'];
      _attachProtoListener = Module['_attachProtoListener'] = wasmExports['Fe'];
      _attachProtoVectorListener = Module['_attachProtoVectorListener'] =
          wasmExports['Ge'];
      _getGraphConfig = Module['_getGraphConfig'] = wasmExports['He'];
      ___getTypeName = wasmExports['Ie'];
      _emwgpuCreateBindGroup = wasmExports['Je'];
      _emwgpuCreateBindGroupLayout = wasmExports['Ke'];
      _emwgpuCreateCommandBuffer = wasmExports['Le'];
      _emwgpuCreateCommandEncoder = wasmExports['Me'];
      _emwgpuCreateComputePassEncoder = wasmExports['Ne'];
      _emwgpuCreateComputePipeline = wasmExports['Oe'];
      _emwgpuCreateExternalTexture = wasmExports['Pe'];
      _emwgpuCreatePipelineLayout = wasmExports['Qe'];
      _emwgpuCreateQuerySet = wasmExports['Re'];
      _emwgpuCreateRenderBundle = wasmExports['Se'];
      _emwgpuCreateRenderBundleEncoder = wasmExports['Te'];
      _emwgpuCreateRenderPassEncoder = wasmExports['Ue'];
      _emwgpuCreateRenderPipeline = wasmExports['Ve'];
      _emwgpuCreateSampler = wasmExports['We'];
      _emwgpuCreateSurface = wasmExports['Xe'];
      _emwgpuCreateTexture = wasmExports['Ye'];
      _emwgpuCreateTextureView = wasmExports['Ze'];
      _emwgpuCreateAdapter = wasmExports['_e'];
      _emwgpuImportBuffer = wasmExports['$e'];
      _emwgpuCreateDevice = wasmExports['af'];
      _emwgpuCreateQueue = wasmExports['bf'];
      _emwgpuCreateShaderModule = wasmExports['cf'];
      _emwgpuOnCreateComputePipelineCompleted = wasmExports['df'];
      _emwgpuOnWorkDoneCompleted = wasmExports['ef'];
      _clearSubgraphs = Module['_clearSubgraphs'] = wasmExports['ff'];
      _pushBinarySubgraph = Module['_pushBinarySubgraph'] = wasmExports['gf'];
      _pushTextSubgraph = Module['_pushTextSubgraph'] = wasmExports['hf'];
      _changeBinaryGraph = Module['_changeBinaryGraph'] = wasmExports['jf'];
      _changeTextGraph = Module['_changeTextGraph'] = wasmExports['kf'];
      _processGl = Module['_processGl'] = wasmExports['lf'];
      _process = Module['_process'] = wasmExports['mf'];
      _bindTextureToCanvas = Module['_bindTextureToCanvas'] = wasmExports['nf'];
      _requestShaderRefreshOnGraphChange =
          Module['_requestShaderRefreshOnGraphChange'] = wasmExports['of'];
      _waitUntilIdle = Module['_waitUntilIdle'] = wasmExports['pf'];
      _closeGraph = Module['_closeGraph'] = wasmExports['qf'];
      _setAutoRenderToScreen = Module['_setAutoRenderToScreen'] =
          wasmExports['rf'];
      _emscripten_builtin_memalign = wasmExports['tf'];
      _memalign = wasmExports['uf'];
      __emscripten_tempret_set = wasmExports['vf'];
      __emscripten_stack_restore = wasmExports['wf'];
      __emscripten_stack_alloc = wasmExports['xf'];
      _emscripten_stack_get_current = wasmExports['yf'];
      dynCall_vii = dynCalls['vii'] = wasmExports['zf'];
      dynCall_viiiiiii = dynCalls['viiiiiii'] = wasmExports['Af'];
      dynCall_viiiiii = dynCalls['viiiiii'] = wasmExports['Bf'];
      dynCall_v = dynCalls['v'] = wasmExports['Cf'];
      dynCall_ii = dynCalls['ii'] = wasmExports['Df'];
      dynCall_vi = dynCalls['vi'] = wasmExports['Ef'];
      dynCall_iiiii = dynCalls['iiiii'] = wasmExports['Ff'];
      dynCall_iii = dynCalls['iii'] = wasmExports['Gf'];
      dynCall_iiii = dynCalls['iiii'] = wasmExports['Hf'];
      dynCall_viiii = dynCalls['viiii'] = wasmExports['If'];
      dynCall_viiiii = dynCalls['viiiii'] = wasmExports['Jf'];
      dynCall_viii = dynCalls['viii'] = wasmExports['Kf'];
      dynCall_iiiijij = dynCalls['iiiijij'] = wasmExports['Lf'];
      dynCall_i = dynCalls['i'] = wasmExports['Mf'];
      dynCall_ji = dynCalls['ji'] = wasmExports['Nf'];
      dynCall_jii = dynCalls['jii'] = wasmExports['Of'];
      dynCall_ff = dynCalls['ff'] = wasmExports['Pf'];
      dynCall_did = dynCalls['did'] = wasmExports['Qf'];
      dynCall_iiiiiii = dynCalls['iiiiiii'] = wasmExports['Rf'];
      dynCall_iiiiii = dynCalls['iiiiii'] = wasmExports['Sf'];
      dynCall_iiiiiiiiiii = dynCalls['iiiiiiiiiii'] = wasmExports['Tf'];
      dynCall_iiiiiiiiiiii = dynCalls['iiiiiiiiiiii'] = wasmExports['Uf'];
      dynCall_iiiiiiiiiiiii = dynCalls['iiiiiiiiiiiii'] = wasmExports['Vf'];
      dynCall_viijj = dynCalls['viijj'] = wasmExports['Wf'];
      dynCall_iiiiiiii = dynCalls['iiiiiiii'] = wasmExports['Xf'];
      dynCall_viji = dynCalls['viji'] = wasmExports['Yf'];
      dynCall_vijiii = dynCalls['vijiii'] = wasmExports['Zf'];
      dynCall_iiiiiiiii = dynCalls['iiiiiiiii'] = wasmExports['_f'];
      dynCall_viiiiiiii = dynCalls['viiiiiiii'] = wasmExports['$f'];
      dynCall_iiiif = dynCalls['iiiif'] = wasmExports['ag'];
      dynCall_viiiiiiiiiiiiiii = dynCalls['viiiiiiiiiiiiiii'] =
          wasmExports['bg'];
      dynCall_viff = dynCalls['viff'] = wasmExports['cg'];
      dynCall_viiffii = dynCalls['viiffii'] = wasmExports['dg'];
      dynCall_viiffi = dynCalls['viiffi'] = wasmExports['eg'];
      dynCall_viiiiiiiiii = dynCalls['viiiiiiiiii'] = wasmExports['fg'];
      dynCall_fi = dynCalls['fi'] = wasmExports['gg'];
      dynCall_viiffff = dynCalls['viiffff'] = wasmExports['hg'];
      dynCall_viiiiij = dynCalls['viiiiij'] = wasmExports['ig'];
      dynCall_vij = dynCalls['vij'] = wasmExports['jg'];
      dynCall_viiiiiffii = dynCalls['viiiiiffii'] = wasmExports['kg'];
      dynCall_vif = dynCalls['vif'] = wasmExports['lg'];
      dynCall_ddd = dynCalls['ddd'] = wasmExports['mg'];
      dynCall_dddd = dynCalls['dddd'] = wasmExports['ng'];
      dynCall_idd = dynCalls['idd'] = wasmExports['og'];
      dynCall_di = dynCalls['di'] = wasmExports['pg'];
      dynCall_jiiiijiiiii = dynCalls['jiiiijiiiii'] = wasmExports['qg'];
      dynCall_viiij = dynCalls['viiij'] = wasmExports['rg'];
      dynCall_viid = dynCalls['viid'] = wasmExports['sg'];
      dynCall_vidd = dynCalls['vidd'] = wasmExports['tg'];
      dynCall_vifd = dynCalls['vifd'] = wasmExports['ug'];
      dynCall_viiiiiiiiiii = dynCalls['viiiiiiiiiii'] = wasmExports['vg'];
      dynCall_iiif = dynCalls['iiif'] = wasmExports['wg'];
      dynCall_jiii = dynCalls['jiii'] = wasmExports['xg'];
      dynCall_iiiiiiiiii = dynCalls['iiiiiiiiii'] = wasmExports['yg'];
      dynCall_iij = dynCalls['iij'] = wasmExports['zg'];
      dynCall_iiiijj = dynCalls['iiiijj'] = wasmExports['Ag'];
      dynCall_viiijjj = dynCalls['viiijjj'] = wasmExports['Bg'];
      dynCall_fff = dynCalls['fff'] = wasmExports['Cg'];
      dynCall_viifii = dynCalls['viifii'] = wasmExports['Dg'];
      dynCall_fii = dynCalls['fii'] = wasmExports['Eg'];
      dynCall_vfiii = dynCalls['vfiii'] = wasmExports['Fg'];
      dynCall_jjj = dynCalls['jjj'] = wasmExports['Gg'];
      dynCall_fiii = dynCalls['fiii'] = wasmExports['Hg'];
      dynCall_viffi = dynCalls['viffi'] = wasmExports['Ig'];
      dynCall_viiiiiiiii = dynCalls['viiiiiiiii'] = wasmExports['Jg'];
      dynCall_viiiffii = dynCalls['viiiffii'] = wasmExports['Kg'];
      dynCall_iiiiiffi = dynCalls['iiiiiffi'] = wasmExports['Lg'];
      dynCall_viiiiiff = dynCalls['viiiiiff'] = wasmExports['Mg'];
      dynCall_viiiiiiiiiiiii = dynCalls['viiiiiiiiiiiii'] = wasmExports['Ng'];
      dynCall_viiiiiiiiiiii = dynCalls['viiiiiiiiiiii'] = wasmExports['Og'];
      dynCall_viiiiiiiiiiiiiiiii = dynCalls['viiiiiiiiiiiiiiiii'] =
          wasmExports['Pg'];
      dynCall_iifff = dynCalls['iifff'] = wasmExports['Qg'];
      dynCall_iiff = dynCalls['iiff'] = wasmExports['Rg'];
      dynCall_iiifiii = dynCalls['iiifiii'] = wasmExports['Sg'];
      dynCall_iiffi = dynCalls['iiffi'] = wasmExports['Tg'];
      dynCall_iiffii = dynCalls['iiffii'] = wasmExports['Ug'];
      dynCall_viiiiiiiiiiiiii = dynCalls['viiiiiiiiiiiiii'] = wasmExports['Vg'];
      dynCall_iif = dynCalls['iif'] = wasmExports['Wg'];
      dynCall_viijii = dynCalls['viijii'] = wasmExports['Xg'];
      dynCall_iiiji = dynCalls['iiiji'] = wasmExports['Yg'];
      dynCall_viiif = dynCalls['viiif'] = wasmExports['Zg'];
      dynCall_viiifiii = dynCalls['viiifiii'] = wasmExports['_g'];
      dynCall_fiif = dynCalls['fiif'] = wasmExports['$g'];
      dynCall_iiiifi = dynCalls['iiiifi'] = wasmExports['ah'];
      dynCall_iiiifiii = dynCalls['iiiifiii'] = wasmExports['bh'];
      dynCall_iiifi = dynCalls['iiifi'] = wasmExports['ch'];
      dynCall_viiiif = dynCalls['viiiif'] = wasmExports['dh'];
      dynCall_viiiifii = dynCalls['viiiifii'] = wasmExports['eh'];
      dynCall_dii = dynCalls['dii'] = wasmExports['fh'];
      dynCall_vifi = dynCalls['vifi'] = wasmExports['gh'];
      dynCall_vidi = dynCalls['vidi'] = wasmExports['hh'];
      dynCall_vijjj = dynCalls['vijjj'] = wasmExports['ih'];
      dynCall_vj = dynCalls['vj'] = wasmExports['jh'];
      dynCall_viij = dynCalls['viij'] = wasmExports['kh'];
      dynCall_jiji = dynCalls['jiji'] = wasmExports['lh'];
      dynCall_iidiiiii = dynCalls['iidiiiii'] = wasmExports['mh'];
      dynCall_iiiiij = dynCalls['iiiiij'] = wasmExports['nh'];
      dynCall_iiiiid = dynCalls['iiiiid'] = wasmExports['oh'];
      dynCall_iiiiijj = dynCalls['iiiiijj'] = wasmExports['ph'];
      dynCall_iiiiiijj = dynCalls['iiiiiijj'] = wasmExports['qh'];
      _asyncify_start_unwind = wasmExports['rh'];
      _asyncify_stop_unwind = wasmExports['sh'];
      _asyncify_start_rewind = wasmExports['th'];
      _asyncify_stop_rewind = wasmExports['uh'];
      memory = wasmMemory = wasmExports['Ec'];
      _kVersionStampBuildChangelistStr =
          Module['_kVersionStampBuildChangelistStr'] =
              (wasmExports['Gc'].value) >>> 0;
      _kVersionStampCitcSnapshotStr = Module['_kVersionStampCitcSnapshotStr'] =
          (wasmExports['Hc'].value) >>> 0;
      _kVersionStampCitcWorkspaceIdStr =
          Module['_kVersionStampCitcWorkspaceIdStr'] =
              (wasmExports['Ic'].value) >>> 0;
      _kVersionStampSourceUriStr = Module['_kVersionStampSourceUriStr'] =
          (wasmExports['Jc'].value) >>> 0;
      _kVersionStampBuildClientStr = Module['_kVersionStampBuildClientStr'] =
          (wasmExports['Kc'].value) >>> 0;
      _kVersionStampBuildClientMintStatusStr =
          Module['_kVersionStampBuildClientMintStatusStr'] =
              (wasmExports['Lc'].value) >>> 0;
      _kVersionStampBuildCompilerStr =
          Module['_kVersionStampBuildCompilerStr'] =
              (wasmExports['Mc'].value) >>> 0;
      _kVersionStampBuildDateTimePstStr =
          Module['_kVersionStampBuildDateTimePstStr'] =
              (wasmExports['Nc'].value) >>> 0;
      _kVersionStampBuildDepotPathStr =
          Module['_kVersionStampBuildDepotPathStr'] =
              (wasmExports['Oc'].value) >>> 0;
      _kVersionStampBuildIdStr = Module['_kVersionStampBuildIdStr'] =
          (wasmExports['Pc'].value) >>> 0;
      _kVersionStampBuildInfoStr = Module['_kVersionStampBuildInfoStr'] =
          (wasmExports['Qc'].value) >>> 0;
      _kVersionStampBuildLabelStr = Module['_kVersionStampBuildLabelStr'] =
          (wasmExports['Rc'].value) >>> 0;
      _kVersionStampBuildTargetStr = Module['_kVersionStampBuildTargetStr'] =
          (wasmExports['Sc'].value) >>> 0;
      _kVersionStampBuildTimestampStr =
          Module['_kVersionStampBuildTimestampStr'] =
              (wasmExports['Tc'].value) >>> 0;
      _kVersionStampBuildToolStr = Module['_kVersionStampBuildToolStr'] =
          (wasmExports['Uc'].value) >>> 0;
      _kVersionStampG3BuildTargetStr =
          Module['_kVersionStampG3BuildTargetStr'] =
              (wasmExports['Vc'].value) >>> 0;
      _kVersionStampVerifiableStr = Module['_kVersionStampVerifiableStr'] =
          (wasmExports['Wc'].value) >>> 0;
      _kVersionStampBuildFdoTypeStr = Module['_kVersionStampBuildFdoTypeStr'] =
          (wasmExports['Xc'].value) >>> 0;
      _kVersionStampBuildBaselineChangelistStr =
          Module['_kVersionStampBuildBaselineChangelistStr'] =
              (wasmExports['Yc'].value) >>> 0;
      _kVersionStampBuildLtoTypeStr = Module['_kVersionStampBuildLtoTypeStr'] =
          (wasmExports['Zc'].value) >>> 0;
      _kVersionStampBuildPropellerTypeStr =
          Module['_kVersionStampBuildPropellerTypeStr'] =
              (wasmExports['_c'].value) >>> 0;
      _kVersionStampBuildPghoTypeStr =
          Module['_kVersionStampBuildPghoTypeStr'] =
              (wasmExports['$c'].value) >>> 0;
      _kVersionStampBuildFdoProfileChangelistStr =
          Module['_kVersionStampBuildFdoProfileChangelistStr'] =
              (wasmExports['ad'].value) >>> 0;
      _kVersionStampBuildMemprofProfileChangelistStr =
          Module['_kVersionStampBuildMemprofProfileChangelistStr'] =
              (wasmExports['bd'].value) >>> 0;
      _kVersionStampBuildUsernameStr =
          Module['_kVersionStampBuildUsernameStr'] =
              (wasmExports['cd'].value) >>> 0;
      _kVersionStampBuildHostnameStr =
          Module['_kVersionStampBuildHostnameStr'] =
              (wasmExports['dd'].value) >>> 0;
      _kVersionStampBuildDirectoryStr =
          Module['_kVersionStampBuildDirectoryStr'] =
              (wasmExports['ed'].value) >>> 0;
      _kVersionStampBuildChangelistInt =
          Module['_kVersionStampBuildChangelistInt'] =
              (wasmExports['fd'].value) >>> 0;
      _kVersionStampCitcSnapshotInt = Module['_kVersionStampCitcSnapshotInt'] =
          (wasmExports['gd'].value) >>> 0;
      _kVersionStampBuildClientMintStatusInt =
          Module['_kVersionStampBuildClientMintStatusInt'] =
              (wasmExports['hd'].value) >>> 0;
      _kVersionStampBuildTimestampInt =
          Module['_kVersionStampBuildTimestampInt'] =
              (wasmExports['id'].value) >>> 0;
      _kVersionStampVerifiableInt = Module['_kVersionStampVerifiableInt'] =
          (wasmExports['jd'].value) >>> 0;
      _kVersionStampBuildCoverageEnabledInt =
          Module['_kVersionStampBuildCoverageEnabledInt'] =
              (wasmExports['kd'].value) >>> 0;
      _kVersionStampBuildBaselineChangelistInt =
          Module['_kVersionStampBuildBaselineChangelistInt'] =
              (wasmExports['ld'].value) >>> 0;
      _kVersionStampPrecookedTimestampStr =
          Module['_kVersionStampPrecookedTimestampStr'] =
              (wasmExports['md'].value) >>> 0;
      _kVersionStampPrecookedClientInfoStr =
          Module['_kVersionStampPrecookedClientInfoStr'] =
              (wasmExports['nd'].value) >>> 0;
      __indirect_function_table = wasmExports['__indirect_function_table'];
      _kVersionStampBuildHasHardeningProtobuf =
          Module['_kVersionStampBuildHasHardeningProtobuf'] =
              (wasmExports['sf'].value) >>> 0;
    }

    var wasmImports = {
      /** @export */ Bc: DefaultErrorReporter,
      /** @export */
      Ac: GetAdapterArchitecture,
      /** @export */
      zc: GetAdapterDescription,
      /** @export */
      yc: GetAdapterDeviceName,
      /** @export */
      xc: GetAdapterVendor,
      /** @export */
      vc: JsGetDeviceMaxSubgroupSize,
      /** @export */
      uc: JsGetDeviceMinSubgroupSize,
      /** @export */
      tc: JsOnEmptyPacketListener,
      /** @export */
      pa: JsOnSimpleListenerBinaryArray,
      /** @export */
      sc: JsOnSimpleListenerBool,
      /** @export */
      rc: JsOnSimpleListenerDouble,
      /** @export */
      qc: JsOnSimpleListenerFloat,
      /** @export */
      pc: JsOnSimpleListenerInt,
      /** @export */
      oc: JsOnSimpleListenerString,
      /** @export */
      nc: JsOnSimpleListenerUint,
      /** @export */
      q: JsOnVectorFinishedListener,
      /** @export */
      mc: JsOnVectorListenerBool,
      /** @export */
      lc: JsOnVectorListenerDouble,
      /** @export */
      kc: JsOnVectorListenerFloat,
      /** @export */
      jc: JsOnVectorListenerInt,
      /** @export */
      ic: JsOnVectorListenerProto,
      /** @export */
      hc: JsOnVectorListenerString,
      /** @export */
      gc: JsOnVectorListenerUint,
      /** @export */
      oa: JsProgressListener,
      /** @export */
      fc: JsWrapErrorListenerInternal,
      /** @export */
      ec: JsWrapLoggingHelpers,
      /** @export */
      g: JsWrapSimpleListeners,
      /** @export */
      dc: ThrowError,
      /** @export */
      na: UseBottomLeftGpuOrigin,
      /** @export */
      cc: __Unwind_RaiseException,
      /** @export */
      a: ___assert_fail,
      /** @export */
      Dc: __asyncjs__CallReadDataFn,
      /** @export */
      Cc: __asyncjs__CallStreamWeightsOnWeb,
      /** @export */
      wc: __asyncjs__InvokeReadDataFn,
      /** @export */
      M: __asyncjs__ReadBufferDataJs,
      /** @export */
      Ga: __asyncjs__mediapipe_map_buffer_jspi,
      /** @export */
      bc: ___syscall_dup,
      /** @export */
      ac: ___syscall_faccessat,
      /** @export */
      m: ___syscall_fcntl64,
      /** @export */
      $b: ___syscall_fstat64,
      /** @export */
      Xa: ___syscall_ftruncate64,
      /** @export */
      _b: ___syscall_getcwd,
      /** @export */
      Zb: ___syscall_getdents64,
      /** @export */
      Yb: ___syscall_getuid32,
      /** @export */
      Xb: ___syscall_ioctl,
      /** @export */
      Wb: ___syscall_lstat64,
      /** @export */
      Vb: ___syscall_mkdirat,
      /** @export */
      Ub: ___syscall_newfstatat,
      /** @export */
      L: ___syscall_openat,
      /** @export */
      Tb: ___syscall_readlinkat,
      /** @export */
      Sb: ___syscall_rmdir,
      /** @export */
      Rb: ___syscall_stat64,
      /** @export */
      ma: ___syscall_unlinkat,
      /** @export */
      Qb: ___syscall_utimensat,
      /** @export */
      Lb: __abort_js,
      /** @export */
      Ta: __embind_register_bigint,
      /** @export */
      Kb: __embind_register_bool,
      /** @export */
      Jb: __embind_register_emval,
      /** @export */
      ka: __embind_register_float,
      /** @export */
      A: __embind_register_function,
      /** @export */
      o: __embind_register_integer,
      /** @export */
      e: __embind_register_memory_view,
      /** @export */
      Ib: __embind_register_std_string,
      /** @export */
      K: __embind_register_std_wstring,
      /** @export */
      Hb: __embind_register_void,
      /** @export */
      ja: __emval_create_invoker,
      /** @export */
      l: __emval_decref,
      /** @export */
      p: __emval_incref,
      /** @export */
      ia: __emval_invoke,
      /** @export */
      ha: __emval_run_destructors,
      /** @export */
      Sa: __gmtime_js,
      /** @export */
      Ra: __localtime_js,
      /** @export */
      Qa: __mktime_js,
      /** @export */
      Pa: __mmap_js,
      /** @export */
      Oa: __munmap_js,
      /** @export */
      Gb: __tzset_js,
      /** @export */
      Wa: _clock_time_get,
      /** @export */
      Fb: custom_emscripten_dbgn,
      /** @export */
      w: _emscripten_asm_const_int,
      /** @export */
      J: _emscripten_errn,
      /** @export */
      Eb: _emscripten_get_heap_max,
      /** @export */
      v: _emscripten_get_now,
      /** @export */
      u: _emscripten_has_asyncify,
      /** @export */
      Db: _emscripten_outn,
      /** @export */
      Cb: _emscripten_pc_get_function,
      /** @export */
      Bb: _emscripten_resize_heap,
      /** @export */
      ga: _emscripten_stack_snapshot,
      /** @export */
      Ab: _emscripten_stack_unwind_buffer,
      /** @export */
      zb: _emscripten_webgl_create_context,
      /** @export */
      yb: _emscripten_webgl_destroy_context,
      /** @export */
      xb: _emscripten_webgl_get_context_attributes,
      /** @export */
      t: _emscripten_webgl_get_current_context,
      /** @export */
      wb: _emscripten_webgl_make_context_current,
      /** @export */
      fa: _emscripten_webgpu_get_device,
      /** @export */
      vb: _emwgpuBufferDestroy,
      /** @export */
      ub: _emwgpuBufferGetMappedRange,
      /** @export */
      tb: _emwgpuBufferUnmap,
      /** @export */
      sb: _emwgpuBufferWriteMappedRange,
      /** @export */
      f: _emwgpuDelete,
      /** @export */
      rb: _emwgpuDeviceCreateBuffer,
      /** @export */
      Na: _emwgpuDeviceCreateComputePipelineAsync,
      /** @export */
      qb: _emwgpuDeviceCreateShaderModule,
      /** @export */
      pb: _emwgpuDeviceDestroy,
      /** @export */
      Ma: _emwgpuQueueOnSubmittedWorkDone,
      /** @export */
      ob: _emwgpuWaitAny,
      /** @export */
      Pb: _environ_get,
      /** @export */
      Ob: _environ_sizes_get,
      /** @export */
      ea: _exit,
      /** @export */
      x: _fd_close,
      /** @export */
      Va: _fd_pread,
      /** @export */
      la: _fd_read,
      /** @export */
      Ua: _fd_seek,
      /** @export */
      B: _fd_write,
      /** @export */
      I: _glActiveTexture,
      /** @export */
      da: _glAttachShader,
      /** @export */
      nb: _glBindAttribLocation,
      /** @export */
      n: _glBindBuffer,
      /** @export */
      z: _glBindFramebuffer,
      /** @export */
      k: _glBindTexture,
      /** @export */
      H: _glBufferData,
      /** @export */
      r: _glClientWaitSync,
      /** @export */
      mb: _glCompileShader,
      /** @export */
      lb: _glCreateProgram,
      /** @export */
      kb: _glCreateShader,
      /** @export */
      ca: _glDeleteFramebuffers,
      /** @export */
      jb: _glDeleteProgram,
      /** @export */
      ba: _glDeleteShader,
      /** @export */
      s: _glDeleteSync,
      /** @export */
      aa: _glDeleteTextures,
      /** @export */
      $: _glDetachShader,
      /** @export */
      _: _glDisableVertexAttribArray,
      /** @export */
      ib: _glDrawArrays,
      /** @export */
      Z: _glEnableVertexAttribArray,
      /** @export */
      Y: _glFenceSync,
      /** @export */
      y: _glFinish,
      /** @export */
      G: _glFramebufferTexture2D,
      /** @export */
      X: _glGenBuffers,
      /** @export */
      hb: _glGenFramebuffers,
      /** @export */
      F: _glGenTextures,
      /** @export */
      E: _glGetError,
      /** @export */
      W: _glGetIntegerv,
      /** @export */
      gb: _glGetProgramiv,
      /** @export */
      fb: _glGetShaderInfoLog,
      /** @export */
      eb: _glGetShaderiv,
      /** @export */
      D: _glGetString,
      /** @export */
      db: _glGetUniformLocation,
      /** @export */
      cb: _glIsTexture,
      /** @export */
      bb: _glLinkProgram,
      /** @export */
      C: _glPixelStorei,
      /** @export */
      V: _glReadPixels,
      /** @export */
      ab: _glShaderSource,
      /** @export */
      $a: _glTexImage2D,
      /** @export */
      j: _glTexParameteri,
      /** @export */
      U: _glTexStorage2D,
      /** @export */
      _a: _glUniform1i,
      /** @export */
      Za: _glUseProgram,
      /** @export */
      T: _glVertexAttribPointer,
      /** @export */
      Ya: _glViewport,
      /** @export */
      S: hardware_concurrency,
      /** @export */
      Nb: _proc_exit,
      /** @export */
      Mb: _random_get,
      /** @export */
      La: _wgpuBufferGetSize,
      /** @export */
      Ka: _wgpuBufferGetUsage,
      /** @export */
      i: _wgpuCommandEncoderBeginComputePass,
      /** @export */
      Ja: _wgpuCommandEncoderCopyBufferToBuffer,
      /** @export */
      R: _wgpuCommandEncoderCopyTextureToBuffer,
      /** @export */
      Fa: _wgpuCommandEncoderCopyTextureToTexture,
      /** @export */
      d: _wgpuCommandEncoderFinish,
      /** @export */
      Ia: _wgpuCommandEncoderResolveQuerySet,
      /** @export */
      Q: _wgpuComputePassEncoderDispatchWorkgroups,
      /** @export */
      h: _wgpuComputePassEncoderEnd,
      /** @export */
      Ea: _wgpuComputePassEncoderSetBindGroup,
      /** @export */
      P: _wgpuComputePassEncoderSetPipeline,
      /** @export */
      Da: _wgpuDeviceCreateBindGroup,
      /** @export */
      Ca: _wgpuDeviceCreateBindGroupLayout,
      /** @export */
      c: _wgpuDeviceCreateCommandEncoder,
      /** @export */
      Ba: _wgpuDeviceCreateComputePipeline,
      /** @export */
      Aa: _wgpuDeviceCreatePipelineLayout,
      /** @export */
      za: _wgpuDeviceCreateQuerySet,
      /** @export */
      ya: _wgpuDeviceCreateTexture,
      /** @export */
      O: _wgpuDeviceGetAdapterInfo,
      /** @export */
      xa: _wgpuDeviceGetLimits,
      /** @export */
      wa: _wgpuDeviceHasFeature,
      /** @export */
      b: _wgpuQueueSubmit,
      /** @export */
      Ha: _wgpuQueueWriteBuffer,
      /** @export */
      va: _wgpuQueueWriteTexture,
      /** @export */
      ua: _wgpuTextureCreateView,
      /** @export */
      N: _wgpuTextureDestroy,
      /** @export */
      ta: _wgpuTextureGetDepthOrArrayLayers,
      /** @export */
      sa: _wgpuTextureGetFormat,
      /** @export */
      ra: _wgpuTextureGetHeight,
      /** @export */
      qa: _wgpuTextureGetWidth
    };

    // Argument name here must shadow the `wasmExports` global so
    // that it is recognised by metadce and minify-import-export-names
    // passes.
    function applySignatureConversions(wasmExports) {
      // First, make a copy of the incoming exports object
      wasmExports = Object.assign({}, wasmExports);
      var makeWrapper_pp = f => a0 => f(a0) >>> 0;
      var makeWrapper_ppp = f => (a0, a1) => f(a0, a1) >>> 0;
      var makeWrapper_p = f => () => f() >>> 0;
      wasmExports['vd'] = makeWrapper_pp(wasmExports['vd']);
      wasmExports['Ie'] = makeWrapper_pp(wasmExports['Ie']);
      wasmExports['tf'] = makeWrapper_ppp(wasmExports['tf']);
      wasmExports['uf'] = makeWrapper_ppp(wasmExports['uf']);
      wasmExports['xf'] = makeWrapper_pp(wasmExports['xf']);
      wasmExports['yf'] = makeWrapper_p(wasmExports['yf']);
      return wasmExports;
    }

    // include: postamble.js
    // === Auto-generated postamble setup entry stuff ===
    async function run() {
      preRun();
      if (runDependencies) {
        await resolveRunDependencies();
      }
      var setStatus = Module['setStatus'];
      if (setStatus) {
        setStatus('Running...');
        // Yield to the event loop to allow the browser to paint "Running..."
        await new Promise(resolve => setTimeout(resolve, 1));
        // Then we want to clear the status text, but only after the rest of
        // this function runs.
        setTimeout(setStatus, 1, '');
      }
      if (ABORT) return;
      initRuntime();
      Module['onRuntimeInitialized']?.();
      postRun();
    }

    var wasmExports;

    // In modularize mode the generated code is within a factory function so we
    // can use await here (since it's not top-level-await).
    wasmExports = await createWasm();

    await run();


    return Module;
  };
})();

// Export using a UMD style export, or ES6 exports if selected
if (typeof exports === 'object' && typeof module === 'object') {
  module.exports = ModuleFactory;
  // This default export looks redundant, but it allows TS to import this
  // commonjs style module.
  module.exports.default = ModuleFactory;
} else if (typeof define === 'function' && define['amd'])
  define([], () => ModuleFactory);
