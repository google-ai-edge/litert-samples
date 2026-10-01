'use strict'; /*

  Copyright The Closure Library Authors.
  SPDX-License-Identifier: Apache-2.0
 */
var m = this || (typeof self !== 'undefined' ? self : {});
function aa(a, b) {
  a = a.split('.');
  for (var c = m, d; a.length && (d = a.shift());)
    a.length || b === void 0 ?
        c[d] && c[d] !== Object.prototype[d] ? c = c[d] : c = c[d] = {} :
        c[d] = b
}
function ba(a, b) {
  a: {
    var c = ['CLOSURE_FLAGS'];
    for (var d = m, e = 0; e < c.length; e++) if (d = d[c[e]], d == null) {
      c = null;
      break a
    } c = d
  } a = c && c[a];
  return a != null ? a : b
}
function ca(a, b, c) {
  return a.call.apply(a.bind, arguments)
}
function da(a, b, c) {
  if (!a) throw Error();
  if (arguments.length > 2) {
    var d = Array.prototype.slice.call(arguments, 2);
    return function() {
      var e = Array.prototype.slice.call(arguments);
      Array.prototype.unshift.apply(e, d);
      return a.apply(b, e)
    }
  }
  return function() {
    return a.apply(b, arguments)
  }
}
function ea(a, b, c) {
  ea = Function.prototype.bind &&
          Function.prototype.bind.toString().indexOf('native code') != -1 ?
      ca :
      da;
  return ea.apply(null, arguments)
}
function ha(a) {
  return a
}
function ia(a, b) {
  function c() {}
  c.prototype = b.prototype;
  a.ya = b.prototype;
  a.prototype = new c;
  a.prototype.constructor = a;
  a.sa = function(d, e, f) {
    for (var g = Array(arguments.length - 2), h = 2; h < arguments.length; h++)
      g[h - 2] = arguments[h];
    return b.prototype[e].apply(d, g)
  }
};
function n(a, b) {
  if (Error.captureStackTrace)
    Error.captureStackTrace(this, n);
  else {
    let c = Error().stack;
    c && (this.stack = c)
  }
  a && (this.message = String(a));
  b !== void 0 && (this.cause = b)
}
ia(n, Error);
n.prototype.name = 'CustomError';
var ja, ka = typeof TextEncoder !== 'undefined';
function la(a) {
  if (ka)
    a = (ja ||= new TextEncoder).encode(a);
  else {
    let c = 0, d = new Uint8Array(3 * a.length);
    for (let e = 0; e < a.length; e++) {
      var b = a.charCodeAt(e);
      if (b < 128)
        d[c++] = b;
      else {
        if (b < 2048)
          d[c++] = b >> 6 | 192;
        else {
          if (b >= 55296 && b <= 57343) {
            if (b <= 56319 && e < a.length) {
              let f = a.charCodeAt(++e);
              if (f >= 56320 && f <= 57343) {
                b = (b - 55296) * 1024 + f - 56320 + 65536;
                d[c++] = b >> 18 | 240;
                d[c++] = b >> 12 & 63 | 128;
                d[c++] = b >> 6 & 63 | 128;
                d[c++] = b & 63 | 128;
                continue
              } else
                e--
            }
            b = 65533
          }
          d[c++] = b >> 12 | 224;
          d[c++] = b >> 6 & 63 | 128
        }
        d[c++] = b & 63 | 128
      }
    }
    a = c === d.length ? d : d.subarray(0, c)
  }
  return a
};
function ma(a) {
  m.setTimeout(() => {
    throw a;
  }, 0)
};
var na = ba(610401301, !1), oa = ba(748402147, !0);
function pa() {
  var a = m.navigator;
  return a && (a = a.userAgent) ? a : ''
}
var qa, ra = m.navigator;
qa = ra ? ra.userAgentData || null : null;
var sa = Array.prototype.some ? function(a, b) {
  return Array.prototype.some.call(a, b, void 0)
} : function(a, b) {
  var c = a.length, d = typeof a === 'string' ? a.split('') : a;
  for (let e = 0; e < c; e++)
    if (e in d && b.call(void 0, d[e], e, a)) return !0;
  return !1
};
var ta = {}, ua = null;
function va(a) {
  var b = a.length, c = b * 3 / 4;
  c % 3 ? c = Math.floor(c) :
          '=.'.indexOf(a[b - 1]) != -1 &&
          (c = '=.'.indexOf(a[b - 2]) != -1 ? c - 2 : c - 1);
  var d = new Uint8Array(c), e = 0;
  wa(a, function(f) {
    d[e++] = f
  });
  return e !== c ? d.subarray(0, e) : d
}
function wa(a, b) {
  function c(e) {
    for (; d < a.length;) {
      let f = a.charAt(d++), g = ua[f];
      if (g != null) return g;
      if (!/^[\s\xa0]*$/.test(f))
        throw Error('Unknown base64 encoding at char: ' + f);
    }
    return e
  }
  Aa();
  for (var d = 0;;) {
    let e = c(-1), f = c(0), g = c(64), h = c(64);
    if (h === 64 && e === -1) break;
    b(e << 2 | f >> 4);
    g != 64 && (b(f << 4 & 240 | g >> 2), h != 64 && b(g << 6 & 192 | h))
  }
}
function Aa() {
  if (!ua) {
    ua = {};
    var a = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
                .split(''),
        b = ['+/=', '+/', '-_=', '-_.', '-_'];
    for (let c = 0; c < 5; c++) {
      let d = a.concat(b[c].split(''));
      ta[c] = d;
      for (let e = 0; e < d.length; e++) {
        let f = d[e];
        ua[f] === void 0 && (ua[f] = e)
      }
    }
  }
};
var Ba = typeof Uint8Array !== 'undefined', Ca;
Ca = na ? !!qa && qa.brands.length > 0 : !1;
var Da =
    !(Ca ? 0 : pa().indexOf('Trident') != -1 || pa().indexOf('MSIE') != -1) &&
    typeof btoa === 'function',
    Ea = /[-_.]/g, Fa = {'-': '+', _: '/', '.': '='};
function Ga(a) {
  return Fa[a] || ''
}
function Ha(a) {
  if (!Da) return va(a);
  a = Ea.test(a) ? a.replace(Ea, Ga) : a;
  a = atob(a);
  var b = new Uint8Array(a.length);
  for (let c = 0; c < a.length; c++) b[c] = a.charCodeAt(c);
  return b
}
function Ia(a) {
  return Ba && a != null && a instanceof Uint8Array
}
var Ja = {};
function Ka() {
  return La ||= new q(null, Ja)
}
var q = class {
  constructor(a, b) {
    Ma(b);
    this.h = a;
    if (a != null && a.length === 0)
      throw Error('ByteString should be constructed with non-empty values');
  }
}
, La;
function Ma(a) {
  if (a !== Ja) throw Error('illegal external caller');
};
function Na(a, b) {
  a.__closure__error__context__984382 ||
      (a.__closure__error__context__984382 = {});
  a.__closure__error__context__984382.severity = b
};
var Oa = void 0;
function Pa(a) {
  a = Error(a);
  Na(a, 'warning');
  return a
}
function Qa(a, b) {
  if (a != null) {
    var c = Oa ??= {};
    var d = c[a] || 0;
    d >= b || (c[a] = d + 1, a = Error(), Na(a, 'incident'), ma(a))
  }
};
function Ra() {
  return typeof BigInt === 'function'
};
var Sa = typeof Symbol === 'function' && typeof Symbol() === 'symbol';
function Ta(a, b, c = !1) {
  return typeof Symbol === 'function' && typeof Symbol() === 'symbol' ?
      c && Symbol.for && a ? Symbol.for(a) :
          a != null        ? Symbol(a) :
                             Symbol() :
      b
}
var Ua = Ta('jas', void 0, !0), Va = Ta(void 0, '1oa'),
    Wa = Ta(void 0, '0ubsb'), Xa = Ta(void 0, '0actk'),
    Ya = Ta('m_m', 'wa', !0);
var Za = {la: {value: 0, configurable: !0, writable: !0, enumerable: !1}},
    $a = Object.defineProperties, r = Sa ? Ua : 'la', ab, bb = [];
v(bb, 7);
ab = Object.freeze(bb);
function cb(a, b) {
  Sa || r in a || $a(a, Za);
  a[r] |= b
}
function v(a, b) {
  Sa || r in a || $a(a, Za);
  a[r] = b
};
var db = {};
function w(a, b) {
  return b === void 0 ? a.h !== eb && !!(2 & (a.v[r] | 0)) :
                        !!(2 & b) && a.h !== eb
}
var eb = {}, fb = Object.freeze({});
function gb(a, b, c) {
  var d = b & 128 ? 0 : -1, e = a.length, f;
  if (f = !!e)
    f = a[e - 1],
    f = f != null && typeof f === 'object' && f.constructor === Object;
  var g = e + (f ? -1 : 0);
  for (b = b & 128 ? 1 : 0; b < g; b++) c(b - d, a[b]);
  if (f) {
    a = a[e - 1];
    for (let h in a) !isNaN(h) && c(+h, a[h])
  }
}
var hb = {};
function ib(a) {
  a.va = !0;
  return a
};
var jb = ib(a => typeof a === 'number'), kb = ib(a => typeof a === 'string'),
    lb = ib(a => typeof a === 'boolean');
var mb = typeof m.BigInt === 'function' && typeof m.BigInt(0) === 'bigint';
var sb =
        ib(a => mb          ? a >= nb && a <= ob :
               a[0] === '-' ? pb(a, qb) :
                              pb(a, rb)),
    qb = Number.MIN_SAFE_INTEGER.toString(),
    nb = mb ? BigInt(Number.MIN_SAFE_INTEGER) : void 0,
    rb = Number.MAX_SAFE_INTEGER.toString(),
    ob = mb ? BigInt(Number.MAX_SAFE_INTEGER) : void 0;
function pb(a, b) {
  if (a.length > b.length) return !1;
  if (a.length < b.length || a === b) return !0;
  for (let c = 0; c < a.length; c++) {
    let d = a[c], e = b[c];
    if (d > e) return !1;
    if (d < e) return !0
  }
};
var x = 0, y = 0, tb;
function ub(a) {
  var b = a >>> 0;
  x = b;
  y = (a - b) / 4294967296 >>> 0
}
function vb(a) {
  if (a < 0) {
    ub(-a);
    let [b, c] = wb(x, y);
    x = b >>> 0;
    y = c >>> 0
  } else
    ub(a)
}
function xb(a, b) {
  var c = b * 4294967296 + (a >>> 0);
  return Number.isSafeInteger(c) ? c : yb(a, b)
}
function yb(a, b) {
  b >>>= 0;
  a >>>= 0;
  if (b <= 2097151)
    var c = '' + (4294967296 * b + a);
  else
    Ra() ? c = '' + (BigInt(b) << BigInt(32) | BigInt(a)) :
           (c = (a >>> 24 | b << 8) & 16777215, b = b >> 16 & 65535,
            a = (a & 16777215) + c * 6777216 + b * 6710656, c += b * 8147497,
            b *= 2, a >= 1E7 && (c += a / 1E7 >>> 0, a %= 1E7),
            c >= 1E7 && (b += c / 1E7 >>> 0, c %= 1E7), c = b + zb(c) + zb(a));
  return c
}
function zb(a) {
  a = String(a);
  return '0000000'.slice(a.length) + a
}
function Ab(a) {
  if (a.length < 16)
    vb(Number(a));
  else if (Ra())
    a = BigInt(a), x = Number(a & BigInt(4294967295)) >>> 0,
    y = Number(a >> BigInt(32) & BigInt(4294967295));
  else {
    let b = +(a[0] === '-');
    y = x = 0;
    let c = a.length;
    for (let d = b, e = (c - b) % 6 + b; e <= c; d = e, e += 6) {
      let f = Number(a.slice(d, e));
      y *= 1E6;
      x = x * 1E6 + f;
      x >= 4294967296 && (y += Math.trunc(x / 4294967296), y >>>= 0, x >>>= 0)
    }
    if (b) {
      let [d, e] = wb(x, y);
      x = d;
      y = e
    }
  }
}
function wb(a, b) {
  b = ~b;
  a ? a = ~a + 1 : b += 1;
  return [a, b]
};
function Bb(a) {
  return Array.prototype.slice.call(a)
};
var Cb = typeof BigInt === 'function' ? BigInt.asIntN : void 0,
    Db = typeof BigInt === 'function' ? BigInt.asUintN : void 0,
    Eb = Number.isSafeInteger, Fb = Number.isFinite, Gb = Math.trunc;
function Hb(a) {
  if (a != null && typeof a !== 'number')
    throw Error(`Value of float/double field must be a number, found ${
        typeof a}: ${a}`);
  return a
}
function Ib(a) {
  if (a == null || typeof a === 'number') return a;
  if (a === 'NaN' || a === 'Infinity' || a === '-Infinity') return Number(a)
}
function Jb(a) {
  if (typeof a !== 'boolean') {
    var b = typeof a;
    throw Error(`Expected boolean but got ${
        b != 'object' ? b :
            a         ? Array.isArray(a) ? 'array' : b :
                        'null'}: ${a}`);
  }
  return a
}
function A(a) {
  if (a == null || typeof a === 'boolean') return a;
  if (typeof a === 'number') return !!a
}
var Kb = /^-?([1-9][0-9]*|0)(\.[0-9]+)?$/;
function Lb(a) {
  switch (typeof a) {
    case 'bigint':
      return !0;
    case 'number':
      return Fb(a);
    case 'string':
      return Kb.test(a);
    default:
      return !1
  }
}
function Mb(a) {
  if (typeof a !== 'number') throw Pa('int32');
  if (!Fb(a)) throw Pa('int32');
  return a | 0
}
function Nb(a) {
  return a == null ? a : Mb(a)
}
function Ob(a) {
  if (a == null) return a;
  if (typeof a === 'string' && a)
    a = +a;
  else if (typeof a !== 'number')
    return;
  return Fb(a) ? a | 0 : void 0
}
function Pb(a) {
  if (a == null) return a;
  if (typeof a === 'string' && a)
    a = +a;
  else if (typeof a !== 'number')
    return;
  return Fb(a) ? a >>> 0 : void 0
}
function Qb(a) {
  if (a == null) return a;
  var b = typeof a;
  if (b === 'bigint') return String(Cb(64, a));
  if (Lb(a)) {
    if (b === 'string') {
      b = Gb(Number(a));
      if (Eb(b))
        a = String(b);
      else if (
          b = a.indexOf('.'), b !== -1 && (a = a.substring(0, b)), b = a.length,
          !(a[0] === '-' ? b < 20 || b === 20 && a <= '-9223372036854775808' :
                           b < 19 || b === 19 && a <= '9223372036854775807'))
        if (Ab(a), a = x, b = y, b & 2147483648)
          if (Ra())
            a = '' + (BigInt(b | 0) << BigInt(32) | BigInt(a >>> 0));
          else {
            let [d, e] = wb(a, b);
            a = '-' + yb(d, e)
          }
        else
          a = yb(a, b);
      return a
    }
    if (b === 'number') {
      a = Gb(a);
      if (!Eb(a)) {
        vb(a);
        b = x;
        var c = y;
        if (a = c & 2147483648)
          b = ~b + 1 >>> 0, c = ~c >>> 0, b == 0 && (c = c + 1 >>> 0);
        b = xb(b, c);
        a = typeof b === 'number' ? a ? -b : b : a ? '-' + b : b
      }
      return a
    }
  }
}
function Rb(a) {
  if (a == null) return a;
  var b = typeof a;
  if (b === 'bigint') return String(Db(64, a));
  if (Lb(a)) {
    if (b === 'string')
      return b = Gb(Number(a)),
             Eb(b) && b >= 0 ?
                 a = String(b) :
                 (b = a.indexOf('.'), b !== -1 && (a = a.substring(0, b)),
                  a[0] === '-' ?
                      b = !1 :
                      (b = a.length,
                       b = b < 20 ? !0 :
                                    b === 20 && a <= '18446744073709551615'),
                  b || (Ab(a), a = yb(x, y))),
             a;
    if (b === 'number')
      return a = Gb(a), a >= 0 && Eb(a) || (vb(a), a = xb(x, y)), a
  }
}
function Sb(a) {
  return a == null || typeof a === 'string' ? a : void 0
}
function Tb(a, b, c) {
  if (a != null && a[Ya] === db) return a;
  if (Array.isArray(a)) {
    var d = a[r] | 0;
    c = d | c & 32 | c & 2;
    c !== d && v(a, c);
    return new b(a)
  }
};
function Ub(a) {
  return a
};
function Vb(a, b, c, d) {
  var e = d !== void 0;
  d = !!d;
  var f = [], g = a.length, h = 4294967295, k = !1, l = !!(b & 64),
      p = l ? b & 128 ? 0 : -1 : void 0;
  if (!(b & 1)) {
    var t = g && a[g - 1];
    t != null && typeof t === 'object' && t.constructor === Object ?
        (g--, h = g) :
        t = void 0;
    !l || b&128 || e || (k = !0, h = (Wb ?? Ub)(h - p, p, a, t, void 0) + p)
  }
  b = void 0;
  for (e = 0; e < g; e++) {
    let u = a[e];
    if (u != null && (u = c(u, d)) != null)
      if (l && e >= h) {
        let z = e - p;
        (b ??= {})[z] = u
      } else
        f[e] = u
  }
  if (t)
    for (let u in t) {
      a = t[u];
      if (a == null || (a = c(a, d)) == null) continue;
      g = +u;
      let z;
      l && !Number.isNaN(g) && (z = g + p) < h ? f[z] = a : (b ??= {})[u] = a
    }
  b && (k ? f.push(b) : f[h] = b);
  return f
}
function Xb(a) {
  switch (typeof a) {
    case 'number':
      return Number.isFinite(a) ? a : '' + a;
    case 'bigint':
      return sb(a) ? Number(a) : '' + a;
    case 'boolean':
      return a ? 1 : 0;
    case 'object':
      if (Array.isArray(a)) {
        var b = a[r] | 0;
        return a.length === 0 && b & 1 ? void 0 : Vb(a, b, Xb)
      }
      if (a != null && a[Ya] === db) return Yb(a);
      if (a instanceof q) {
        b = a.h;
        if (b == null)
          a = '';
        else if (typeof b === 'string')
          a = b;
        else {
          if (Da) {
            for (var c = '', d = 0, e = b.length - 10240; d < e;)
              c += String.fromCharCode.apply(null, b.subarray(d, d += 10240));
            c += String.fromCharCode.apply(null, d ? b.subarray(d) : b);
            b = btoa(c)
          } else {
            c === void 0 && (c = 0);
            Aa();
            c = ta[c];
            d = Array(Math.floor(b.length / 3));
            e = c[64] || '';
            let l = 0, p = 0;
            for (; l < b.length - 2; l += 3) {
              var f = b[l], g = b[l + 1], h = b[l + 2], k = c[f >> 2];
              f = c[(f & 3) << 4 | g >> 4];
              g = c[(g & 15) << 2 | h >> 6];
              h = c[h & 63];
              d[p++] = k + f + g + h
            }
            k = 0;
            h = e;
            switch (b.length - l) {
              case 2:
                k = b[l + 1], h = c[(k & 15) << 2] || e;
              case 1:
                b = b[l], d[p] = c[b >> 2] + c[(b & 3) << 4 | k >> 4] + h + e
            }
            b = d.join('')
          }
          a = a.h = b
        }
        return a
      }
      return
  }
  return a
}
var Wb;
function Yb(a) {
  a = a.v;
  return Vb(a, a[r] | 0, Xb)
};
var Zb, $b;
function ac(a) {
  switch (typeof a) {
    case 'boolean':
      return Zb ||= [0, void 0, !0];
    case 'number':
      return a > 0 ? void 0 : a === 0 ? $b ||= [0, void 0] : [-a, void 0];
    case 'string':
      return [0, a];
    case 'object':
      return a
  }
}
function bc(a, b, c, d = 0) {
  if (a == null) {
    var e = 32;
    c ? (a = [c], e |= 128) : a = [];
    b && (e = e & -16760833 | (b & 1023) << 14)
  } else {
    if (!Array.isArray(a)) throw Error('narr');
    e = a[r] | 0;
    if (oa && 1 & e) throw Error('rfarr');
    2048&e && !(2 & e) && cc();
    if (e & 256) throw Error('farr');
    if (e & 64) return (e | d) !== e && v(a, e | d), a;
    if (c && (e |= 128, c !== a[0])) throw Error('mid');
    a: {
      c = a; e |= 64; var f = c.length; if (f) {
        var g = f - 1;
        let k = c[g];
        if (k != null && typeof k === 'object' && k.constructor === Object) {
          b = e & 128 ? 0 : -1;
          g -= b;
          if (g >= 1024) throw Error('pvtlmt');
          for (var h in k) f = +h, f < g && (c[f + b] = k[h], delete k[h]);
          e = e & -16760833 | (g & 1023) << 14;
          break a
        }
      } if (b) {
        h = Math.max(b, f - (e & 128 ? 0 : -1));
        if (h > 1024) throw Error('spvt');
        e = e & -16760833 | (h & 1023) << 14
      }
    }
  }
  v(a, e | 64 | d);
  return a
}
function cc() {
  if (oa) throw Error('carr');
  Qa(Xa, 5)
};
function dc(a, b) {
  if (typeof a !== 'object') return a;
  if (Array.isArray(a)) {
    var c = a[r] | 0;
    a.length === 0 && c&1 ? a = void 0 :
                            c & 2 ||
            (!b || 4096 & c || 16 & c ? a = ec(a, c, !1, b && !(c & 16)) :
                                        (cb(a, 34), c & 4 && Object.freeze(a)));
    return a
  }
  if (a != null && a[Ya] === db)
    return b = a.v, c = b[r] | 0,
           w(a, c)         ? a :
               fc(a, b, c) ? hc(a, b) :
                             ec(b, c);
  if (a instanceof q) return a
}
function hc(a, b, c) {
  a = new a.constructor(b);
  c && (a.h = eb);
  a.m = eb;
  return a
}
function ec(a, b, c, d) {
  d ??= !!(34 & b);
  a = Vb(a, b, dc, d);
  d = 32;
  c && (d |= 2);
  b = b & 16769217 | d;
  v(a, b);
  return a
}
function ic(a) {
  if (a.h !== eb) return !1;
  var b = a.v;
  b = ec(b, b[r] | 0);
  cb(b, 2048);
  a.v = b;
  a.h = void 0;
  a.m = void 0;
  return !0
}
function jc(a) {
  if (!ic(a) && w(a, a.v[r] | 0)) throw Error();
}
function kc(a, b) {
  b === void 0 && (b = a[r] | 0);
  b&32 && !(b & 4096) && v(a, b | 4096)
}
function fc(a, b, c) {
  return c & 2 ? !0 : c & 32 && !(c & 4096) ? (v(b, c | 2), a.h = eb, !0) : !1
};
function B(a, b, c) {
  a = lc(a.v, b, void 0, c);
  if (a !== null) return a
}
function lc(a, b, c, d) {
  if (b === -1) return null;
  var e = b + (c ? 0 : -1), f = a.length - 1;
  if (!(f < 1 + (c ? 0 : -1))) {
    if (e >= f) {
      var g = a[f];
      if (g != null && typeof g === 'object' && g.constructor === Object) {
        c = g[b];
        var h = !0
      } else if (e === f)
        c = g;
      else
        return
    } else
      c = a[e];
    if (d && c != null) {
      d = d(c);
      if (d == null) return d;
      if (!Object.is(d, c)) return h ? g[b] = d : a[e] = d, d
    }
    return c
  }
}
function mc(a, b, c) {
  jc(a);
  a = a.v;
  C(a, a[r] | 0, b, c)
}
function C(a, b, c, d, e) {
  var f = c + (e ? 0 : -1), g = a.length - 1;
  if (g >= 1 + (e ? 0 : -1) && f >= g) {
    let h = a[g];
    if (h != null && typeof h === 'object' && h.constructor === Object)
      return h[c] = d, b
  }
  if (f <= g) return a[f] = d, b;
  d !== void 0 &&
      (g = (b ??= a[r] | 0) >> 14 & 1023 || 536870912,
       c >= g ? d != null && (a[g + (e ? 0 : -1)] = {[c]: d}) : a[f] = d);
  return b
}
function nc(a, b, c, d, e) {
  var f = a.v, g = f[r] | 0;
  d = w(a, g) ? 1 : d;
  e = !!e || d === 3;
  d === 2 && ic(a) && (f = a.v, g = f[r] | 0);
  a = oc(f, b);
  var h = a === ab ? 7 : a[r] | 0, k = pc(h, g);
  var l = 4 & k ? !1 : !0;
  if (l) {
    4&k && (a = Bb(a), h = 0, k = qc(k, g), g = C(f, g, b, a));
    let p = 0, t = 0;
    for (; p < a.length; p++) {
      let u = c(a[p]);
      u != null && (a[t++] = u)
    }
    t < p && (a.length = t);
    c = (k | 4) & -513;
    k = c &= -1025;
    k &= -4097
  }
  k !== h && (v(a, k), 2 & k && Object.freeze(a));
  return a = rc(a, k, f, g, b, d, l, e)
}
function rc(a, b, c, d, e, f, g, h) {
  var k = b;
  f === 1 || (f !== 4 ? 0 : 2 & b || !(16 & b) && 32 & d) ?
      sc(b) ||
          (b |=
           !a.length || g && !(4096 & b) || 32 & d && !(4096 & b || 16 & b) ?
               2 :
               256,
           b !== k && v(a, b), Object.freeze(a)) :
      (f === 2 && sc(b) && (a = Bb(a), k = 0, b = qc(b, d), d = C(c, d, e, a)),
       sc(b) || (h || (b |= 16), b !== k && v(a, b)));
  2&b || !(4096 & b || 16 & b) || kc(c, d);
  return a
}
function oc(a, b, c) {
  a = lc(a, b, c);
  return Array.isArray(a) ? a : ab
}
function pc(a, b) {
  2&b && (a |= 2);
  return a | 1
}
function sc(a) {
  return !!(2 & a) && !!(4 & a) || !!(256 & a)
}
function tc(a, b, c) {
  jc(a);
  a = a.v;
  var d = a[r] | 0;
  if (c == null)
    C(a, d, b);
  else {
    var e = c === ab ? 7 : c[r] | 0, f = e, g = sc(e),
        h = g || Object.isFrozen(c);
    g || (e = 0);
    h || (c = Bb(c), f = 0, e = qc(e, d), h = !1);
    e |= 5;
    e |= (4 & e ? 512 & e ? 512 : 1024 & e ? 1024 : 0 : void 0) ?? 1024;
    for (g = 0; g < c.length; g++) {
      let k = c[g], l = Mb(k);
      Object.is(k, l) ||
          (h && (c = Bb(c), f = 0, e = qc(e, d), h = !1), c[g] = l)
    }
    e !== f && (h && (c = Bb(c), e = qc(e, d)), v(c, e));
    C(a, d, b, c)
  }
}
function uc(a, b, c, d) {
  jc(a);
  a = a.v;
  C(a, a[r] | 0, b, (d === '0' ? Number(c) === 0 : c === d) ? void 0 : c)
}
function vc(a) {
  if (Sa) return a[Va] ?? (a[Va] = new Map);
  if (Va in a) return a[Va];
  var b = new Map;
  Object.defineProperty(a, Va, {value: b});
  return b
}
function wc(a, b, c) {
  var d = xc, e = a.get(d);
  if (e != null) return e;
  e = 0;
  for (let f = 0; f < d.length; f++) {
    let g = d[f];
    lc(b, g) != null && (e !== 0 && (c = C(b, c, e)), e = g)
  }
  a.set(d, e);
  return e
}
function yc(a, b, c, d) {
  var e = !1;
  d = lc(a, d, void 0, f => {
    var g = Tb(f, c, b);
    e = g !== f && g != null;
    return g
  });
  if (d != null) return e && !w(d) && kc(a, b), d
}
function zc(a, b, c) {
  var d = a.v, e = d[r] | 0;
  b = yc(d, e, b, c);
  if (b == null) return b;
  e = d[r] | 0;
  if (!w(a, e)) {
    var f, g = b;
    let h = g.v, k = h[r] | 0;
    w(g, k) ?
        fc(g, h, k) ? f = hc(g, h, !0) : f = new g.constructor(ec(h, k, !1)) :
        f = g;
    f !== b &&
        (ic(a) && (d = a.v, e = d[r] | 0), b = f, e = C(d, e, c, b), kc(d, e))
  }
  return b
}
function Ac(a) {
  a == null && (a = void 0);
  return a
}
function Bc(a, b, c) {
  c = Ac(c);
  mc(a, b, c);
  c && !w(c) && kc(a.v);
  return a
}
function qc(a, b) {
  return a = (2 & b ? a | 2 : a & -3) & -273
}
function Cc(a, b, c, d) {
  var e = d;
  jc(a);
  var f = d = a.v, g = d[r] | 0, h = w(a, g) ? 1 : 2;
  h === 2 && ic(a) && (f = a.v, g = f[r] | 0);
  a = oc(f, b);
  var k = a === ab ? 7 : a[r] | 0, l = pc(k, g), p = !(4 & l);
  if (p) {
    var t = a, u = g;
    let z = !!(2 & l);
    z && (u |= 2);
    let T = !z, xa = !0, fa = 0, ya = 0;
    for (; fa < t.length; fa++) {
      let Y = Tb(t[fa], c, u);
      if (Y instanceof c) {
        if (!z) {
          let za = w(Y);
          T &&= !za;
          xa &&= za
        }
        t[ya++] = Y
      }
    }
    ya < fa && (t.length = ya);
    l |= 4;
    l = xa ? l & -4097 : l | 4096;
    l = T ? l | 8 : l & -9
  }
  l !== k && (v(a, l), 2 & l && Object.freeze(a));
  b = a = rc(a, l, f, g, b, h, p, !0);
  e = e != null ? e : new c;
  b.push(e);
  f = c = b === ab ? 7 : b[r] | 0;
  (e = w(e)) ? (c &= -9, b.length === 1 && (c &= -4097)) : c |= 4096;
  c !== f && v(b, c);
  e || kc(d)
}
function D(a, b) {
  return Pb(B(a, b)) ?? 0
}
function Dc(a, b, c) {
  mc(a, b, c == null ? c : Jb(c))
}
function Ec(a, b, c) {
  uc(a, b, c == null ? c : Jb(c), !1)
}
function E(a, b, c) {
  uc(a, b, Nb(c), 0)
}
function Fc(a, b, c) {
  if (c != null) {
    if (typeof c !== 'number') throw Pa('uint32');
    if (!Fb(c)) throw Pa('uint32');
    c >>>= 0
  }
  mc(a, b, c)
}
function F(a, b, c) {
  if (c != null && typeof c !== 'string') throw Error();
  uc(a, b, c, '')
}
function G(a, b, c) {
  jc(a);
  a = nc(a, b, Sb, 2, !0);
  b = a.push;
  if (typeof c !== 'string') throw Error();
  b.call(a, c)
};
var Gc = class {
  constructor(a, b, c) {
    this.buffer = a;
    if (c && !b) throw Error();
  }
};
function Hc(a) {
  if (typeof a === 'string') return new Gc(Ha(a), !0);
  if (Array.isArray(a)) return new Gc(new Uint8Array(a), !0);
  if (a.constructor === Uint8Array) return new Gc(a, !1);
  if (a.constructor === ArrayBuffer)
    return a = new Uint8Array(a), new Gc(a, !1);
  if (a.constructor === q) {
    Ma(Ja);
    var b = a.h;
    b = b == null || Ia(b) ? b : typeof b === 'string' ? Ha(b) : null;
    b = (b == null ? b : a.h = b) || new Uint8Array(0);
    return new Gc(b, !0, a)
  }
  if (a instanceof Uint8Array)
    return a = a.constructor === Uint8Array ?
               a :
               new Uint8Array(a.buffer, a.byteOffset, a.byteLength),
           new Gc(a, !1);
  throw Error();
};
function Ic(a) {
  return new Jc(a & 4294967295, Math.floor(a / 4294967296))
}
function Kc(a) {
  if (!a) return Lc ||= new Jc(0, 0);
  if (!/^\d+$/.test(a)) return null;
  Ab(a);
  return new Jc(x, y)
}
var Jc = class {
  constructor(a, b) {
    this.j = a >>> 0;
    this.h = b >>> 0
  }
}
, Lc;
function Mc(a) {
  return new Nc(a & 4294967295, Math.floor(a / 4294967296))
}
function Oc(a) {
  if (!a) return Pc ||= new Nc(0, 0);
  if (!/^-?\d+$/.test(a)) return null;
  Ab(a);
  return new Nc(x, y)
}
var Nc = class {
  constructor(a, b) {
    this.j = a >>> 0;
    this.h = b >>> 0
  }
}
, Pc, Qc, Rc, Sc, Tc, Uc, Vc, Wc;
function Xc(a, b, c) {
  if (typeof BigInt64Array !== 'undefined')
    return Vc ||
               (Vc = new BigInt64Array(1), Wc = new Uint32Array(Vc.buffer),
                Vc[0] = BigInt(1), Uc = Wc[0] === 1),
           Vc[0] = a, a = Uc ? 0 : 1, new b(Wc[a], Wc[1 - a]);
  Tc ||
      (Qc = BigInt(Number.MIN_SAFE_INTEGER),
       Rc = BigInt(Number.MAX_SAFE_INTEGER), Sc = BigInt(4294967295),
       Tc = BigInt(32));
  if (a >= Qc && a <= Rc) return c(Number(a));
  a = BigInt.asUintN(64, a);
  return new b(Number(a & Sc), Number(a >> Tc))
};
function Yc(a, b, c) {
  for (; c > 0 || b > 127;)
    a.h.push(b & 127 | 128), b = (b >>> 7 | c << 25) >>> 0, c >>>= 7;
  a.h.push(b)
}
function Zc(a, b) {
  for (; b > 127;) a.h.push(b & 127 | 128), b >>>= 7;
  a.h.push(b)
}
function $c(a, b) {
  if (b >= 0)
    Zc(a, b);
  else {
    for (let c = 0; c < 9; c++) a.h.push(b & 127 | 128), b >>= 7;
    a.h.push(1)
  }
}
var ad = class {
  constructor() {
    this.h = []
  }
  length() {
    return this.h.length
  }
  end() {
    var a = this.h;
    this.h = [];
    return a
  }
};
function bd(a, b) {
  b.length !== 0 && (a.l.push(b), a.j += b.length)
}
function H(a, b, c) {
  Zc(a.h, b * 8 + c)
}
function cd(a, b) {
  H(a, b, 2);
  b = a.h.end();
  bd(a, b);
  b.push(a.j);
  return b
}
function dd(a, b) {
  var c = b.pop();
  for (c = a.j + a.h.length() - c; c > 127;)
    b.push(c & 127 | 128), c >>>= 7, a.j++;
  b.push(c);
  a.j++
}
function ed(a, b, c) {
  H(a, b, 2);
  Zc(a.h, c.length);
  bd(a, a.h.end());
  bd(a, c)
}
var fd = class {
  constructor() {
    this.l = [];
    this.j = 0;
    this.h = new ad
  }
};
function I() {
  var a = class {
    constructor() {
      throw Error();
    }
  };
  Object.setPrototypeOf(a, a.prototype);
  return a
}
var gd = I(), hd = I(), id = I(), jd = I(), kd = I(), ld = I(), md = I(),
    nd = I(), od = I(), pd = I();
var J = class {
  constructor(a, b) {
    this.v = bc(a, b, void 0, 2048)
  }
  toJSON() {
    var a = Yb(this);
    return a
  }
};
J.prototype[Ya] = db;
J.prototype.toString = function() {
  return this.v.toString()
};
var K = class {
  constructor(a, b) {
    this.h = a;
    a = ha(gd);
    this.j = !!a && b === a || !1
  }
};
function qd(a, b, c, d, e) {
  b = rd(b, d);
  b != null && (c = cd(a, c), e(b, a), dd(a, c))
}
var sd = new K(qd, gd), td = new K(qd, gd), ud = Symbol(), vd = Symbol(), wd,
    xd;
function yd(a) {
  var b = zd, c = Ad, d = a[ud];
  if (d) return d;
  d = {};
  d.ta = a;
  d.X = ac(a[0]);
  var e = a[1], f = 1;
  e && e.constructor === Object &&
      (d.fa = e, e = a[++f],
       typeof e === 'function' &&
           (d.ma = !0, wd ??= e, xd ??= a[f + 1], e = a[f += 2]));
  for (var g = {}; e && Array.isArray(e) && e.length &&
       typeof e[0] === 'number' && e[0] > 0;) {
    for (var h = 0; h < e.length; h++) g[e[h]] = e;
    e = a[++f]
  }
  for (h = 1; e !== void 0;) {
    typeof e === 'number' && (h += e, e = a[++f]);
    let p;
    var k = void 0;
    e instanceof K ? p = e : (p = sd, f--);
    if (p?.j) {
      e = a[++f];
      k = a;
      var l = f;
      typeof e === 'function' && (e = e(), k[l] = e);
      k = e
    }
    e = a[++f];
    l = h + 1;
    typeof e === 'number' && e < 0 && (l -= e, e = a[++f]);
    for (; h < l; h++) {
      let t = g[h];
      k ? c(d, h, p, k, t) : b(d, h, p, t)
    }
  }
  return a[ud] = d
}
function rd(a, b) {
  if (a instanceof J) return a.v;
  if (Array.isArray(a)) return bc(a, b[0], b[1])
};
function zd(a, b, c) {
  a[b] = c.h
}
function Ad(a, b, c, d) {
  var e, f, g = c.h;
  a[b] = (h, k, l) => g(h, k, l, f ||= yd(d).X, e ||= Bd(d))
}
function Bd(a) {
  var b = a[vd];
  if (!b) {
    let c = yd(a);
    b = (d, e) => Cd(d, e, c);
    a[vd] = b
  }
  return b
}
function Cd(a, b, c) {
  gb(a, a[r] | 0, (d, e) => {
    if (e != null) {
      var f = Dd(c, d);
      f ? f(b, e, d) : d < 500 || Qa(Wa, 3)
    }
  })
}
function Dd(a, b) {
  var c = a[b];
  if (c) return c;
  if (c = a.fa)
    if (c = c[b]) {
      c = Array.isArray(c) ? c[0] instanceof K ? c : [td, c] : [c, void 0];
      var d = c[0].h;
      if (c = c[1]) {
        let e = Bd(c), f = yd(c).X;
        c = a.ma ? xd(f, e) : (g, h, k) => d(g, h, k, f, e)
      } else
        c = d;
      return a[b] = c
    }
};
var Ed = 0, Fd = Ed;
if (kb(Fd)) {
  if (!/^\s*(?:-?[1-9]\d*|0)?\s*$/.test(Fd)) throw Error(String(Fd));
} else {
  var Gd;
  if (Gd = jb(Fd)) Gd = !Number.isSafeInteger(Fd);
  if (Gd) throw Error(String(Fd));
}
mb || (Ed = lb(Ed) ? Ed ? '1' : '0' : kb(Ed) ? Ed.trim() || '0' : String(Ed));
function Hd(a, b) {
  if (Array.isArray(b)) {
    var c = b[r] | 0;
    if (c & 4) return b;
    for (var d = 0, e = 0; d < b.length; d++) {
      let f = a(b[d]);
      f != null && (b[e++] = f)
    }
    e < d && (b.length = e);
    a = (c | 5) & -1537;
    a !== c && v(b, a);
    a&2 && Object.freeze(b);
    return b
  }
}
function L(a, b) {
  return new K(a, b)
}
function Id(a, b, c) {
  b = Ib(b);
  b != null &&
      (H(a, c, 5), a = a.h, c = tb ||= new DataView(new ArrayBuffer(8)),
       c.setFloat32(0, +b, !0), y = 0, b = x = c.getUint32(0, !0),
       a.h.push(b >>> 0 & 255), a.h.push(b >>> 8 & 255),
       a.h.push(b >>> 16 & 255), a.h.push(b >>> 24 & 255))
}
function Jd(a, b, c) {
  b = Ob(b);
  b != null && b != null && (H(a, c, 0), $c(a.h, b))
}
function Kd(a, b, c) {
  b = A(b);
  b != null && (H(a, c, 0), a.h.h.push(b ? 1 : 0))
}
function Ld(a, b, c) {
  b = Sb(b);
  b != null && ed(a, c, la(b))
}
function Md(a, b, c, d, e) {
  b = rd(b, d);
  b != null && (c = cd(a, c), e(b, a), dd(a, c))
}
function Nd(a, b, c) {
  b = Pb(b);
  b != null && b != null && (H(a, c, 0), Zc(a.h, b))
}
function Od(a, b, c) {
  b = Ob(b);
  b != null && (b = parseInt(b, 10), H(a, c, 0), $c(a.h, b))
}
var Pd = L(Id, nd), Qd = L(Id, nd),
    Rd = L(function(a, b, c) {
      b = Qb(b);
      if (b != null) {
        switch (typeof b) {
          case 'string':
            Oc(b)
        }
        if (b != null) switch (H(a, c, 0), typeof b) {
            case 'number':
              a = a.h;
              vb(b);
              Yc(a, x, y);
              break;
            case 'bigint':
              c = Xc(b, Nc, Mc);
              Yc(a.h, c.j, c.h);
              break;
            default:
              c = Oc(b), Yc(a.h, c.j, c.h)
          }
      }
    }, ld), Sd = L(function(a, b, c) {
              b = Rb(b);
              if (b != null) {
                switch (typeof b) {
                  case 'string':
                    Kc(b)
                }
                if (b != null) switch (H(a, c, 0), typeof b) {
                    case 'number':
                      a = a.h;
                      vb(b);
                      Yc(a, x, y);
                      break;
                    case 'bigint':
                      c = Xc(b, Jc, Ic);
                      Yc(a.h, c.j, c.h);
                      break;
                    default:
                      c = Kc(b), Yc(a.h, c.j, c.h)
                  }
              }
            }, md), M = L(Jd, jd), Td;
Td = new K(function(a, b, c) {
  b = Hd(Ob, b);
  if (b != null && b.length) {
    c = cd(a, c);
    for (let d = 0; d < b.length; d++) $c(a.h, b[d]);
    dd(a, c)
  }
}, jd);
var N = L(Jd, jd), Ud = L(Jd, jd), O = L(Kd, hd), P = L(Kd, hd), Q = L(Ld, id),
    R;
R = new K(function(a, b, c) {
  b = Hd(Sb, b);
  if (b != null)
    for (let g = 0; g < b.length; g++) {
      var d = a, e = c, f = b[g];
      f != null && ed(d, e, la(f))
    }
}, id);
var S = L(Ld, id), Vd = L(Ld, id),
    U =
        function(a, b, c = gd) {
      return new K(b, c)
    }(
            function(a, b, c, d, e) {
              if (a.h() !== 2) return !1;
              var f = a.j;
              d = bc(void 0, d[0], d[1]);
              var g = b[r] | 0;
              if (g & 2) throw Error();
              var h = g & 128 ? hb : void 0, k = oc(b, c, h),
                  l = k === ab ? 7 : k[r] | 0, p = pc(l, g);
              if (2 & p || sc(p) || 16 & p)
                p === l || sc(p) || v(k, p), k = Bb(k), l = 0, p = qc(p, g),
                                             C(b, g, c, k, h);
              p &= -13;
              p !== l && v(k, p);
              k.push(d);
              f.call(a, d, e);
              return !0
            },
            function(a, b, c, d, e) {
              if (Array.isArray(b)) {
                for (let f = 0; f < b.length; f++) Md(a, b[f], c, d, e);
                a = b[r] | 0;
                a&1 || v(b, a | 1)
              }
            }),
    V = new K(Md, gd), Wd = L(Nd, kd), Xd = L(Nd, kd), W = L(Od, pd), Yd;
Yd = new K(function(a, b, c) {
  b = Hd(Ob, b);
  if (b != null && b.length) {
    c = cd(a, c);
    for (let d = 0; d < b.length; d++) $c(a.h, b[d]);
    dd(a, c)
  }
}, pd);
var Zd = L(Od, pd);
function $d(a) {
  return function() {
    var b = new fd;
    Cd(this.v, b, yd(a));
    bd(b, b.h.end());
    var c = new Uint8Array(b.j), d = b.l, e = d.length, f = 0;
    for (let g = 0; g < e; g++) {
      let h = d[g];
      c.set(h, f);
      f += h.length
    }
    b.l = [c];
    return c
  }
};
function ae(a, b) {
  if (b != null)
    if (Array.isArray(b))
      mc(a, 2, Vb(b, 0, Xb));
    else if (typeof b === 'string' || b instanceof q || Ia(b)) {
      if (b != null)
        if (typeof b === 'string')
          b = b ? new q(b, Ja) : Ka();
        else if (b.constructor !== q)
          if (Ia(b))
            b = b.length ? new q(new Uint8Array(b), Ja) : Ka();
          else
            throw Error();
      uc(a, 2, b, Ka())
    } else
      throw Error(
          'invalid value in Any.value field: ' + b +
          ' expected a ByteString, a base64 encoded string, a Uint8Array or a jspb array');
}
var be = class extends J {
  constructor(a) {
    super(a)
  }
};
var ce = [
  0, S,
  L(
      function(a, b, c) {
        if (b != null) {
          if (b instanceof J) {
            let d = b.xa;
            d ? (b = d(b), b != null && ed(a, c, Hc(b).buffer)) : Qa(Wa, 3);
            return
          }
          if (Array.isArray(b)) {
            Qa(Wa, 3);
            return
          }
        }
        b = b == null || typeof b == 'string' || b instanceof q ? b : void 0;
        b != null && ed(a, c, Hc(b).buffer)
      },
      od)
];
var de = typeof AsyncContext !== 'undefined' && typeof AsyncContext.Snapshot ===
        'function' ?
    a => a && AsyncContext.Snapshot.wrap(a) :
    a => a;
function ee(a, b) {
  a.m(b);
  a.j < 100 && (a.j++, b.next = a.h, a.h = b)
}
class fe {
  constructor(a, b) {
    this.l = a;
    this.m = b;
    this.j = 0;
    this.h = null
  }
  get() {
    if (this.j > 0) {
      this.j--;
      var a = this.h;
      this.h = a.next;
      a.next = null
    } else
      a = this.l();
    return a
  }
};
function ge() {
  var a = he, b = null;
  a.h && (b = a.h, a.h = a.h.next, a.h || (a.j = null), b.next = null);
  return b
}
class ie {
  constructor() {
    this.j = this.h = null
  }
  add(a, b) {
    var c = je.get();
    c.set(a, b);
    this.j ? this.j.next = c : this.h = c;
    this.j = c
  }
}
var je = new fe(() => new ke, a => a.reset());
class ke {
  constructor() {
    this.next = this.h = this.j = null
  }
  set(a, b) {
    this.j = a;
    this.h = b;
    this.next = null
  }
  reset() {
    this.next = this.h = this.j = null
  }
};
var le, me = !1, he = new ie, oe = (a, b) => {
  le || ne();
  me || (le(), me = !0);
  he.add(a, b)
}, ne = () => {
  var a = Promise.resolve(void 0);
  le = () => {
    a.then(pe)
  }
};
function pe() {
  for (var a; a = ge();) {
    try {
      a.j.call(a.h)
    } catch (b) {
      ma(b)
    }
    ee(je, a)
  }
  me = !1
};
function qe(a, b) {
  for (let c in a) b.call(void 0, a[c], c, a)
}
var re =
    'constructor hasOwnProperty isPrototypeOf propertyIsEnumerable toLocaleString toString valueOf'
        .split(' ');
function se(a, b) {
  for (let e = 1; e < arguments.length; e++) {
    var c = arguments[e];
    for (d in c) a[d] = c[d];
    for (let f = 0; f < re.length; f++) {
      var d = re[f];
      Object.prototype.hasOwnProperty.call(c, d) && (a[d] = c[d])
    }
  }
}; /*

Copyright Google LLC
SPDX-License-Identifier: Apache-2.0
*/
var te = globalThis.trustedTypes, ue;
function ve() {
  var a = null;
  if (!te) return a;
  try {
    let b = c => c;
    a = te.createPolicy(
        'goog#html', {createHTML: b, createScript: b, createScriptURL: b})
  } catch (b) {
  }
  return a
};
var we = class {
  constructor(a) {
    this.h = a
  }
  toString() {
    return this.h + ''
  }
};
function xe(a) {
  var b;
  ue === void 0 && (ue = ve());
  a = (b = ue) ? b.createScriptURL(a) : a;
  return new we(a)
}
function ye(a) {
  if (a instanceof we) return a.h;
  throw Error('');
};
function ze(a = document) {
  a = a.querySelector?.('script[nonce]');
  return a == null ? '' : a.nonce || a.getAttribute('nonce') || ''
};
function Ae(a, b) {
  a.src = ye(b);
  (b = ze(a.ownerDocument)) && a.setAttribute('nonce', b)
};
function Be(...a) {
  self.importScripts(...a.map(b => ye(b)))
};
function Ce(a, ...b) {
  if (b.length === 0) return xe(a[0]);
  var c = a[0];
  for (let d = 0; d < b.length; d++) c += encodeURIComponent(b[d]) + a[d + 1];
  return xe(c)
}
function De(a, b) {
  var c = ye(a).toString();
  var d = c.split(/[?#]/);
  a = /[?]/.test(c) ? '?' + d[1] : '';
  var e = d[0];
  c = /[#]/.test(c) ? '#' + (a ? d[2] : d[1]) : '';
  b = e + (e.slice(-1) === '/' ? '' : '/') + encodeURIComponent(b);
  return xe(b + a + c)
};
function Ee(a, b) {
  qe(b, function(c, d) {
    d == 'style'             ? a.style.cssText = c :
        d == 'class'         ? a.className = c :
        d == 'for'           ? a.htmlFor = c :
        Fe.hasOwnProperty(d) ? a.setAttribute(Fe[d], c) :
        d.lastIndexOf('aria-', 0) == 0 || d.lastIndexOf('data-', 0) == 0 ?
                               a.setAttribute(d, c) :
                               a[d] = c
  })
}
var Fe = {
  cellpadding: 'cellPadding',
  cellspacing: 'cellSpacing',
  colspan: 'colSpan',
  frameborder: 'frameBorder',
  height: 'height',
  maxlength: 'maxLength',
  nonce: 'nonce',
  role: 'role',
  rowspan: 'rowSpan',
  type: 'type',
  usemap: 'useMap',
  valign: 'vAlign',
  width: 'width'
};
function Ge(a) {
  this.h = a || m.document || document
}
function He(a) {
  a = a.h;
  var b = 'SCRIPT';
  a.contentType === 'application/xhtml+xml' && (b = b.toLowerCase());
  return a.createElement(b)
};
function Ie() {};
function Je(a) {
  if (!a) return !1;
  try {
    return !!a.$goog_Thenable
  } catch (b) {
    return !1
  }
};
function Ke(a) {
  this.h = 0;
  this.B = void 0;
  this.m = this.j = this.l = null;
  this.u = this.A = !1;
  if (a != Ie) try {
      let b = this;
      a.call(
          void 0,
          function(c) {
            Le(b, 2, c)
          },
          function(c) {
            Le(b, 3, c)
          })
    } catch (b) {
      Le(this, 3, b)
    }
}
function Me() {
  this.next = this.l = this.j = this.m = this.h = null;
  this.u = !1
}
Me.prototype.reset = function() {
  this.l = this.j = this.m = this.h = null;
  this.u = !1
};
var Ne = new fe(
    function() {
      return new Me
    },
    function(a) {
      a.reset()
    });
function Oe(a, b, c) {
  var d = Ne.get();
  d.m = a;
  d.j = b;
  d.l = c;
  return d
}
Ke.prototype.then = function(a, b, c) {
  return Pe(
      this, de(typeof a === 'function' ? a : null),
      de(typeof b === 'function' ? b : null), c)
};
Ke.prototype.$goog_Thenable = !0;
Ke.prototype.cancel = function(a) {
  if (this.h == 0) {
    let b = new Qe(a);
    oe(function() {
      Re(this, b)
    }, this)
  }
};
function Re(a, b) {
  if (a.h == 0)
    if (a.l) {
      var c = a.l;
      if (c.j) {
        var d = 0, e = null, f = null;
        for (let g = c.j;
             g && (g.u || (d++, g.h == a && (e = g), !(e && d > 1)));
             g = g.next)
          e || (f = g);
        e &&
            (c.h == 0 && d == 1 ? Re(c, b) :
                                  (f ? (d = f, d.next == c.m && (c.m = d),
                                        d.next = d.next.next) :
                                       Se(c),
                                   Te(c, e, 3, b)))
      }
      a.l = null
    } else
      Le(a, 3, b)
}
function Ue(a, b) {
  a.j || a.h != 2 && a.h != 3 || Ve(a);
  a.m ? a.m.next = b : a.j = b;
  a.m = b
}
function Pe(a, b, c, d) {
  var e = Oe(null, null, null);
  e.h = new Ke(function(f, g) {
    e.m = b ? function(h) {
      try {
        let k = b.call(d, h);
        f(k)
      } catch (k) {
        g(k)
      }
    } : f;
    e.j = c ? function(h) {
      try {
        let k = c.call(d, h);
        k === void 0 && h instanceof Qe ? g(h) : f(k)
      } catch (k) {
        g(k)
      }
    } : g
  });
  e.h.l = a;
  Ue(a, e);
  return e.h
}
Ke.prototype.C = function(a) {
  this.h = 0;
  Le(this, 2, a)
};
Ke.prototype.F = function(a) {
  this.h = 0;
  Le(this, 3, a)
};
function Le(a, b, c) {
  if (a.h == 0) {
    a === c && (b = 3, c = new TypeError('Promise cannot resolve to itself'));
    a.h = 1;
    a: {
      var d = c, e = a.C, f = a.F; if (d instanceof Ke) {
        Ue(d, Oe(e || Ie, f || null, a));
        var g = !0
      } else if (Je(d)) d.then(e, f, a),
                                   g = !0;
      else {
        var h = typeof d;
        if (h == 'object' && d != null || h == 'function') try {
            let k = d.then;
            if (typeof k === 'function') {
              We(d, k, e, f, a);
              g = !0;
              break a
            }
          } catch (k) {
            f.call(a, k);
            g = !0;
            break a
          }
        g = !1
      }
    } g ||
        (a.B = c, a.h = b, a.l = null, Ve(a),
         b != 3 || c instanceof Qe || Xe(a, c))
  }
}
function We(a, b, c, d, e) {
  function f(k) {
    h || (h = !0, d.call(e, k))
  }
  function g(k) {
    h || (h = !0, c.call(e, k))
  }
  var h = !1;
  try {
    b.call(a, g, f)
  } catch (k) {
    f(k)
  }
}
function Ve(a) {
  a.A || (a.A = !0, oe(a.D, a))
}
function Se(a) {
  var b = null;
  a.j && (b = a.j, a.j = b.next, b.next = null);
  a.j || (a.m = null);
  return b
}
Ke.prototype.D = function() {
  for (var a; a = Se(this);) Te(this, a, this.h, this.B);
  this.A = !1
};
function Te(a, b, c, d) {
  if (c == 3 && b.j && !b.u)
    for (; a && a.u; a = a.l) a.u = !1;
  if (b.h)
    b.h.l = null, Ye(b, c, d);
  else
    try {
      b.u ? b.m.call(b.l) : Ye(b, c, d)
    } catch (e) {
      Ze.call(null, e)
    }
  ee(Ne, b)
}
function Ye(a, b, c) {
  b == 2 ? a.m.call(a.l, c) : a.j && a.j.call(a.l, c)
}
function Xe(a, b) {
  a.u = !0;
  oe(function() {
    a.u && Ze.call(null, b)
  })
}
var Ze = ma;
function Qe(a) {
  n.call(this, a)
}
ia(Qe, n);
Qe.prototype.name = 'cancel'; /*

Copyright 2005, 2007 Bob Ippolito. All Rights Reserved.
Copyright The Closure Library Authors.
SPDX-License-Identifier: MIT
*/
function $e(a) {
  var b = af;
  this.u = [];
  this.H = b;
  this.G = a || null;
  this.m = this.l = !1;
  this.j = void 0;
  this.C = this.I = this.B = !1;
  this.A = 0;
  this.h = null;
  this.D = 0
}
$e.prototype.cancel = function(a) {
  if (this.l)
    this.j instanceof $e && this.j.cancel();
  else {
    if (this.h) {
      let b = this.h;
      delete this.h;
      a ? b.cancel(a) : (b.D--, b.D <= 0 && b.cancel())
    }
    this.H ? this.H.call(this.G, this) : this.C = !0;
    this.l || (a = new bf(this), cf(this), df(this, !1, a))
  }
};
$e.prototype.F = function(a, b) {
  this.B = !1;
  df(this, a, b)
};
function df(a, b, c) {
  a.l = !0;
  a.j = c;
  a.m = !b;
  ef(a)
}
function cf(a) {
  if (a.l) {
    if (!a.C) throw new ff(a);
    a.C = !1
  }
}
function gf(a, b, c, d) {
  var e = a.l;
  e || (b === c ? b = c = de(b) : (b = de(b), c = de(c)));
  a.u.push([b, c, d]);
  e && ef(a)
}
$e.prototype.then = function(a, b, c) {
  var d, e, f = new Ke(function(g, h) {
              e = g;
              d = h
            });
  gf(this, e, function(g) {
    g instanceof bf ? f.cancel() : d(g);
    return hf
  }, this);
  return f.then(a, b, c)
};
$e.prototype.$goog_Thenable = !0;
function jf(a) {
  return sa(a.u, function(b) {
    return typeof b[1] === 'function'
  })
}
var hf = {};
function ef(a) {
  if (a.A && a.l && jf(a)) {
    var b = a.A, c = kf[b];
    c && (m.clearTimeout(c.h), delete kf[b]);
    a.A = 0
  }
  a.h && (a.h.D--, delete a.h);
  b = a.j;
  for (var d = c = !1; a.u.length && !a.B;) {
    var e = a.u.shift(), f = e[0];
    let h = e[1];
    e = e[2];
    if (f = a.m ? h : f) try {
        var g = f.call(e || a.G, b);
        g === hf && (g = void 0);
        g !== void 0 &&
            (a.m = a.m && (g == b || g instanceof Error), a.j = b = g);
        if (Je(b) || typeof m.Promise === 'function' && b instanceof m.Promise)
          d = !0, a.B = !0
      } catch (k) {
        b = k, a.m = !0, jf(a) || (c = !0)
      }
  }
  a.j = b;
  d &&
      (g = ea(a.F, a, !0), d = ea(a.F, a, !1),
       b instanceof $e ? (gf(b, g, d), b.I = !0) : b.then(g, d));
  c && (b = new lf(b), kf[b.h] = b, a.A = b.h)
}
function ff() {
  n.call(this)
}
ia(ff, n);
ff.prototype.message = 'Deferred has already fired';
ff.prototype.name = 'AlreadyCalledError';
function bf() {
  n.call(this)
}
ia(bf, n);
bf.prototype.message = 'Deferred was canceled';
bf.prototype.name = 'CanceledError';
function lf(a) {
  this.h = m.setTimeout(ea(this.l, this), 0);
  this.j = a
}
lf.prototype.l = function() {
  delete kf[this.h];
  throw this.j;
};
var kf = {};
function mf(a) {
  var b = {}, c = b.document || document, d = ye(a).toString(),
      e = He(new Ge(c)), f = {Z: e, aa: void 0}, g = new $e(f), h = null,
      k = b.timeout != null ? b.timeout : 5E3;
  k > 0 && (h = window.setTimeout(function() {
    nf(e, !0);
    var l = new of(1, 'Timeout reached for loading script ' + d);
    cf(g);
    df(g, !1, l)
  }, k), f.aa = h);
  e.onload = e.onreadystatechange = function() {
    e.readyState && e.readyState != 'loaded' && e.readyState != 'complete' ||
        (nf(e, b.ua || !1, h), cf(g), df(g, !0, null))
  };
  e.onerror = function() {
    nf(e, !0, h);
    var l = new of(0, 'Error while loading script ' + d);
    cf(g);
    df(g, !1, l)
  };
  f = b.attributes || {};
  se(f, {type: 'text/javascript', charset: 'UTF-8'});
  Ee(e, f);
  Ae(e, a);
  pf(c).appendChild(e);
  return g
}
function pf(a) {
  var b;
  return (b = (a || document).getElementsByTagName('HEAD')) && b.length !== 0 ?
      b[0] :
      a.documentElement
}
function af() {
  if (this && this.Z) {
    let a = this.Z;
    a && a.tagName == 'SCRIPT' && nf(a, !0, this.aa)
  }
}
function nf(a, b, c) {
  c != null && m.clearTimeout(c);
  a.onload = () => {};
  a.onerror = () => {};
  a.onreadystatechange = () => {};
  b && window.setTimeout(function() {
    a && a.parentNode && a.parentNode.removeChild(a)
  }, 0)
}
function of (a, b) {
  var c = 'Jsloader error (code #' + a + ')';
  b && (c += ': ' + b);
  n.call(this, c);
  this.code = a
}
ia(of, n);
var qf = [0, W, -1, O];
var rf = {};
rf[336783863]=[0,Q,O,-1,M,[0,[1,2,3,4,5,6,7,8,9],V,[0],V,[0,O,Q,O,W,-1,Yd,Q,-1,[0,O,-1],W,O,-1,qf],V,[0,Q,-2],V,[0,M,O,1,O,-4],V,[0,M,W,O,-1,Td,W,-1,O,-1],V,[0,Q,-2],V,[0,Q,W],V,[0,3,O,-1,2,[0,[2],M,V,[0,Wd]],[0,W,O,W,O,W,4,[0,O,Q,-1,O]],[0,[3,4],Q,-1,V,[0,M],V,[0,W,-1]],[0]],V,qf],[0,Q],O,[0,[1,3],[2,4],V,[0,Td],-1,V,[0,R],-1,U,[0,Q,-1]],Q];
var sf = class extends J {
  constructor(a) {
    super(a)
  }
};
var tf = [0, Rd, -1, P, -3, Rd, Td, S, N, Rd, -1, P, N, P, -2, S];
var uf = class extends J {
  constructor(a) {
    super(a, 500)
  }
  N(a) {
    return Bc(this, 7, a)
  }
};
var vf = [-1, {}];
var wf = [0, Q, 1, vf];
var xf = [0, Q, R, vf];
function yf(a, b) {
  Cc(a, 1, uf, b)
}
var zf = class extends J {
  constructor(a) {
    super(a, 500)
  }
  N(a) {
    return Bc(this, 1001, a)
  }
};
zf.prototype.j = $d([
  -500,
  U,
  [
    -500, S,  -1, R,         -3, [-2, rf, O], U, ce, N,   -1,
    wf,   xf, U,  [0, S, P], S,  tf,          N, R,  987, R
  ],
  4,
  U,
  [-500, Q, -1, [-1, {}], 998, Q],
  U,
  [-500, Q, R, -1, [-2, {}, O], 997, R, -1],
  N,
  U,
  [-500, Q, R, vf, 998, R],
  R,
  N,
  wf,
  xf,
  U,
  [0, S, -1, vf],
  R,
  -2,
  tf,
  S,
  -1,
  P,
  [0, P, Xd],
  978,
  vf,
  U,
  ce
]);
var Af = class extends J {
  constructor(a) {
    super(a)
  }
};
var Bf, Cf = new Uint8Array([
          0, 97, 115, 109, 1,  0, 0, 0, 1,  5, 1,   96, 0,   1,  123, 3,
          2, 1,  0,   10,  10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11
        ]);
async function Df(a) {
  if (a) return !0;
  if (Bf === void 0) try {
      await WebAssembly.instantiate(Cf), Bf = !0
    } catch {
      Bf = !1
    }
  return Bf
}
async function Ef(a, b, c) {
  var d = await Df(c) ? '' : '_nosimd';
  c = `wasm${c ? '_module' : ''}${d}_internal`;
  return {
    wasmLoaderPath: De(b, `${a}_${c}.js`),
        wasmBinaryPath: De(b, `${a}_${c}.wasm`)
  }
}
var Ff = class {};
Ff.forVisionTasks = function(a, b = !1) {
  return Ef('vision', a ?? Ce``, b)
};
Ff.forTextTasks = function(a, b = !1) {
  return Ef('text', a ?? Ce``, b)
};
Ff.forGenAiTasks = function(a, b = !1) {
  return Ef('genai', a ?? Ce``, b)
};
Ff.forAudioTasks = function(a, b = !1) {
  return Ef('audio', a ?? Ce``, b)
};
Ff.isSimdSupported = function(a = !1) {
  return Df(a)
};
export {Ff as FilesetResolver};
var Gf = class {
  close() {}
};
function Hf(a) {
  function b(g, h) {
    return new ReadableStream({
      start() {},
      async pull(k) {
        e = e.then(async () => {
          if (g.cache.length > 0)
            k.enqueue(g.cache.shift());
          else {
            var {value: l, done: p} = await a.read();
            l && (h.active && h.cache.push(l), g.active && k.enqueue(l));
            p && k.close()
          }
        });
        await e
      },
      cancel() {
        g.active = !1;
        g.cache.length = 0;
        h.active || a.cancel()
      }
    })
  }
  var c = {cache: [], active: !0}, d = {cache: [], active: !0},
      e = Promise.resolve(), f = b(c, d);
  c = b(d, c);
  return [f.getReader(), c.getReader()]
}
async function If(a, b) {
  for (var c = new Uint8Array(b), d = 0; d < b;) {
    let {value: e, done: f} = await a.read();
    if (e) {
      let g = e.subarray(0, b - d);
      c.set(g, d);
      d += g.length
    }
    if (f)
      throw Error(
          `Expected ${b} bytes, but stream ended after reading ${d} bytes.`);
  }
  await a.cancel();
  return c
}
async function Jf(a) {
  for (var b = [], c = 0;;) {
    let {done: d, value: e} = await a.read();
    if (d) break;
    b.push(e);
    c += e.length
  }
  if (b.length === 0) return new Uint8Array(0);
  if (b.length === 1) return b[0];
  a = new Uint8Array(c);
  c = 0;
  for (let d of b) a.set(d, c), c += d.length;
  return a
}
var Kf = [
  [
    0,
    async a => {
      var b = (new TextEncoder).encode('TFL3').length;
      a = await If(a, b + 4);
      return (new TextDecoder('utf-8')).decode(a.subarray(4, b + 4)) === 'TFL3'
    }
  ],
  [
    1,
    async a => {
      a = await If(a, 6);
      return a[4] === 80 && a[5] === 75
    }
  ],
  [
    2,
    async a => {
      a = await If(a, 8);
      return (new TextDecoder('utf-8')).decode(a) === 'LITERTLM'
    }
  ]
];
async function Lf(a, b) {
  var c = new Uint8Array(b), d = 0;
  if (a.h) {
    var e = Math.min(b, a.h.length);
    c.set(a.h.subarray(0, e), 0);
    d += e;
    d < a.h.length ? a.h = a.h.subarray(d) : a.h = void 0
  }
  for (; d < b;) {
    let {value: f, done: g} = await a.stream.read();
    f &&
        (e = f.subarray(0, b - d), c.set(e, d),
         f.length > e.length && (a.h = f.subarray(b - d)), d += e.length);
    if (g)
      throw Error(
          `Expected ${b} bytes, but stream ended after reading ${d} bytes.`);
  }
  return c
}
class Mf {
  constructor(a) {
    this.stream = a;
    this.h = void 0;
    this.closed = this.stream.closed
  }
  async read() {
    if (this.h) {
      let a = this.h.slice();
      this.h = void 0;
      return {
        value: a, done: !1
      }
    }
    return this.stream.read()
  }
  cancel(a) {
    return this.stream.cancel(a)
  }
  releaseLock() {
    return this.stream.releaseLock()
  }
}
async function Nf(a, b) {
  a = new Mf(a);
  var c = await Lf(a, 32);
  c = new DataView(c.buffer);
  c = Number(c.getBigUint64(24, !0));
  var d = await Lf(a, c - 32);
  b = b(d);
  if (b < 0)
    throw Error(
        '.litertlm file could not be read or did not contain a web-formatted LLM');
  await Lf(a, b - c);
  return a
}
async function Of(a) {
  var b = [], c;
  for (let [e, f] of Kf) {
    let g = e;
    var d = f;
    [a, c] = Hf(a);
    d = await d(c);
    await c.cancel();
    d && b.push(g)
  }
  await a.cancel();
  if (b.length === 0) throw Error('No model format matched.');
  if (b.length === 1) return b[0];
  throw Error(`Multiple model formats matched: ${b}`);
}
function Pf(a) {
  return new ReadableStream({
    start() {},
    async pull(b) {
      b.enqueue(a);
      b.close()
    }
  })
};
function Qf(a = navigator) {
  a = a.userAgent;
  return a.includes('Safari') && !a.includes('Chrome')
}
function Rf() {
  var a = navigator;
  return typeof OffscreenCanvas === 'undefined' ? !1 :
      Qf(a) ? (a = a.userAgent.match(/Version\/([\d]+).*Safari/)) &&
              a.length >= 1 && Number(a[1]) >= 17 ?
              !0 :
              !1 :
              !0
};
async function Sf(a) {
  if (typeof importScripts === 'function')
    Be(a);
  else
    return mf(a)
};
var Tf = async (a, b, c) => {
  var d = X;
  a && await Sf(a);
  if (!self.ModuleFactory) throw Error('ModuleFactory not set.');
  if (b && (await Sf(b), !self.ModuleFactory))
    throw Error('ModuleFactory not set.');
  self.Module && c &&
      (a = self.Module, a.locateFile = c.locateFile,
       c.mainScriptUrlOrBlob &&
           (a.mainScriptUrlOrBlob = c.mainScriptUrlOrBlob));
  c = await self.ModuleFactory(self.Module || c);
  self.ModuleFactory = self.Module = void 0;
  return new d(c, null)
};
function Z(a, b, c) {
  a.m ||
      console.error(
          'No wasm multistream support detected: ensure dependency inclusion of :gl_graph_runner_internal_multi_input target');
  b = a.i.stringToNewUTF8(b);
  c(b);
  a.i._free(b)
}
function Uf(a, b, c) {
  a.m ||
      console.error(
          'No wasm multistream support detected: ensure dependency inclusion of :gl_graph_runner_internal_multi_input target');
  var d = new Uint32Array(b.length);
  for (let e = 0; e < b.length; e++) d[e] = a.i.stringToNewUTF8(b[e]);
  b = a.i._malloc(d.length * 4);
  a.i.HEAPU32.set(d, b >> 2);
  c(b);
  for (let e of d) a.i._free(e);
  a.i._free(b)
}
function Vf(a, b, c) {
  a.i.simpleListeners = a.i.simpleListeners || {};
  a.i.simpleListeners[b] = c
}
function Wf(a, b, c) {
  var d = [];
  a.i.simpleListeners = a.i.simpleListeners || {};
  a.i.simpleListeners[b] = (e, f, g) => {
    f ? (c(d, g), d = []) : d.push(e)
  }
}
var Xf =
        function(a) {
  return class extends a {
    get T() {
      return this.i
    }
    ja() {
      if (typeof this.T._mediapipeLoggerGetEncodedApiKey === 'function') {
        let b = this.T._mediapipeLoggerGetEncodedApiKey();
        this.T._decodeBase64(b)
      }
    }
  }
}(function(a) {
  return class extends a {
    pa() {
      this.i._registerModelResourcesGraphService()
    }
  }
}(class {
  constructor(a, b) {
    this.l = !0;
    this.i = a;
    this.h = null;
    this.j = 0;
    this.m = typeof this.i._addIntToInputStream === 'function';
    b !== void 0 ?
        this.i.canvas = b :
        Rf() ?
        this.i.canvas = new OffscreenCanvas(1, 1) :
        (console.warn(
             'OffscreenCanvas not supported and GraphRunner constructor glCanvas parameter is undefined. Creating backup canvas.'),
         this.i.canvas = document.createElement('canvas'))
  }
  async initializeGraph(a) {
    var b = await (await fetch(a)).arrayBuffer();
    a = !(a.endsWith('.pbtxt') || a.endsWith('.textproto'));
    this.setGraph(new Uint8Array(b), a)
  }
  setGraphFromString(a) {
    this.setGraph((new TextEncoder).encode(a), !1)
  }
  setGraph(a, b) {
    var c = a.length, d = this.i._malloc(c);
    this.i.HEAPU8.set(a, d);
    b ? this.i._changeBinaryGraph(c, d) : this.i._changeTextGraph(c, d);
    this.i._free(d)
  }
  configureAudio(a, b, c, d, e) {
    this.i._configureAudio ||
        console.warn(
            'Attempting to use configureAudio without support for input audio. Is build dep ":gl_graph_runner_audio" missing?');
    Z(this, d || 'input_audio', f => {
      e = e || 'audio_header';
      Z(this, e, g => {this.i._configureAudio(f, g, a, b ?? 0, c)})
    })
  }
  setAutoResizeCanvas(a) {
    this.l = a
  }
  setAutoRenderToScreen(a) {
    this.i._setAutoRenderToScreen(a)
  }
  setGpuBufferVerticalFlip(a) {
    this.i.gpuOriginForWebTexturesIsBottomLeft = a
  }
  attachErrorListener(a) {
    this.i.errorListener = a
  }
  attachEmptyPacketListener(a, b) {
    this.i.emptyPacketListeners = this.i.emptyPacketListeners || {};
    this.i.emptyPacketListeners[a] = b
  }
  addAudioToStream(a, b, c) {
    this.addAudioToStreamWithShape(a, 0, 0, b, c)
  }
  addAudioToStreamWithShape(a, b, c, d, e) {
    var f = a.length * 4;
    this.j !== f &&
        (this.h && this.i._free(this.h), this.h = this.i._malloc(f),
         this.j = f);
    this.i.HEAPF32.set(a, this.h / 4);
    Z(this, d, g => {this.i._addAudioToInputStream(this.h, b, c, g, e)})
  }
  addGpuBufferToStream(a, b, c) {
    Z(this, b, d => {
      if (!this.i.canvas) throw Error('No OpenGL canvas configured.');
      d ? this.i._bindTextureToStream(d) : this.i._bindTextureToCanvas();
      var e = this.i.canvas.getContext('webgl2') ||
          this.i.canvas.getContext('webgl');
      if (!e)
        throw Error(
            'Failed to obtain WebGL context from the provided canvas. `getContext()` should only be invoked with `webgl` or `webgl2`.');
      this.i.gpuOriginForWebTexturesIsBottomLeft &&
          e.pixelStorei(e.UNPACK_FLIP_Y_WEBGL, !0);
      e.texImage2D(e.TEXTURE_2D, 0, e.RGBA, e.RGBA, e.UNSIGNED_BYTE, a);
      this.i.gpuOriginForWebTexturesIsBottomLeft &&
          e.pixelStorei(e.UNPACK_FLIP_Y_WEBGL, !1);
      var [f, g] = a.videoWidth !== void 0 ? [a.videoWidth, a.videoHeight] :
          a.naturalWidth !== void 0        ? [a.naturalWidth, a.naturalHeight] :
          a.displayWidth !== void 0        ? [a.displayWidth, a.displayHeight] :
                                             [a.width, a.height];
      !this.l || f === this.i.canvas.width && g === this.i.canvas.height ||
          (this.i.canvas.width = f, this.i.canvas.height = g);
      var [h, k] = [f, g];
      this.i._addBoundTextureToStream(d, h, k, c)
    })
  }
  addBoolToStream(a, b, c) {
    Z(this, b, d => {this.i._addBoolToInputStream(a, d, c)})
  }
  addDoubleToStream(a, b, c) {
    Z(this, b, d => {this.i._addDoubleToInputStream(a, d, c)})
  }
  addFloatToStream(a, b, c) {
    Z(this, b, d => {this.i._addFloatToInputStream(a, d, c)})
  }
  addIntToStream(a, b, c) {
    Z(this, b, d => {this.i._addIntToInputStream(a, d, c)})
  }
  addUintToStream(a, b, c) {
    Z(this, b, d => {this.i._addUintToInputStream(a, d, c)})
  }
  addStringToStream(a, b, c) {
    Z(this, b,
      d => {Z(this, a, e => {this.i._addStringToInputStream(e, d, c)})})
  }
  addStringRecordToStream(a, b, c) {
    Z(this, b,
      d => {
          Uf(this, Object.keys(a),
             e => {
                 Uf(this, Object.values(a),
                    f => {this.i._addFlatHashMapToInputStream(
                        e, f, Object.keys(a).length, d, c)})})})
  }
  addProtoToStream(a, b, c, d) {
    Z(this, c, e => {Z(this, b, f => {
                 var g = this.i._malloc(a.length);
                 this.i.HEAPU8.set(a, g);
                 this.i._addProtoToInputStream(g, a.length, f, e, d);
                 this.i._free(g)
               })})
  }
  addEmptyPacketToStream(a, b) {
    Z(this, a, c => {this.i._addEmptyPacketToInputStream(c, b)})
  }
  addBoolVectorToStream(a, b, c) {
    Z(this, b, d => {
      var e = this.i._allocateBoolVector(a.length);
      if (!e) throw Error('Unable to allocate new bool vector on heap.');
      for (let f of a) this.i._addBoolVectorEntry(e, f);
      this.i._addBoolVectorToInputStream(e, d, c)
    })
  }
  addDoubleVectorToStream(a, b, c) {
    Z(this, b, d => {
      var e = this.i._allocateDoubleVector(a.length);
      if (!e) throw Error('Unable to allocate new double vector on heap.');
      for (let f of a) this.i._addDoubleVectorEntry(e, f);
      this.i._addDoubleVectorToInputStream(e, d, c)
    })
  }
  addFloatVectorToStream(a, b, c) {
    Z(this, b, d => {
      var e = this.i._allocateFloatVector(a.length);
      if (!e) throw Error('Unable to allocate new float vector on heap.');
      for (let f of a) this.i._addFloatVectorEntry(e, f);
      this.i._addFloatVectorToInputStream(e, d, c)
    })
  }
  addIntVectorToStream(a, b, c) {
    Z(this, b, d => {
      var e = this.i._allocateIntVector(a.length);
      if (!e) throw Error('Unable to allocate new int vector on heap.');
      for (let f of a) this.i._addIntVectorEntry(e, f);
      this.i._addIntVectorToInputStream(e, d, c)
    })
  }
  addUintVectorToStream(a, b, c) {
    Z(this, b, d => {
      var e = this.i._allocateUintVector(a.length);
      if (!e)
        throw Error('Unable to allocate new unsigned int vector on heap.');
      for (let f of a) this.i._addUintVectorEntry(e, f);
      this.i._addUintVectorToInputStream(e, d, c)
    })
  }
  addStringVectorToStream(a, b, c) {
    Z(this, b, d => {
      var e = this.i._allocateStringVector(a.length);
      if (!e) throw Error('Unable to allocate new string vector on heap.');
      for (let f of a) Z(this, f, g => {this.i._addStringVectorEntry(e, g)});
      this.i._addStringVectorToInputStream(e, d, c)
    })
  }
  addBoolToInputSidePacket(a, b) {
    Z(this, b, c => {this.i._addBoolToInputSidePacket(a, c)})
  }
  addDoubleToInputSidePacket(a, b) {
    Z(this, b, c => {this.i._addDoubleToInputSidePacket(a, c)})
  }
  addFloatToInputSidePacket(a, b) {
    Z(this, b, c => {this.i._addFloatToInputSidePacket(a, c)})
  }
  addIntToInputSidePacket(a, b) {
    Z(this, b, c => {this.i._addIntToInputSidePacket(a, c)})
  }
  addUintToInputSidePacket(a, b) {
    Z(this, b, c => {this.i._addUintToInputSidePacket(a, c)})
  }
  addStringToInputSidePacket(a, b) {
    Z(this, b,
      c => {Z(this, a, d => {this.i._addStringToInputSidePacket(d, c)})})
  }
  addProtoToInputSidePacket(a, b, c) {
    Z(this, c, d => {Z(this, b, e => {
                 var f = this.i._malloc(a.length);
                 this.i.HEAPU8.set(a, f);
                 this.i._addProtoToInputSidePacket(f, a.length, e, d);
                 this.i._free(f)
               })})
  }
  addBoolVectorToInputSidePacket(a, b) {
    Z(this, b, c => {
      var d = this.i._allocateBoolVector(a.length);
      if (!d) throw Error('Unable to allocate new bool vector on heap.');
      for (let e of a) this.i._addBoolVectorEntry(d, e);
      this.i._addBoolVectorToInputSidePacket(d, c)
    })
  }
  addDoubleVectorToInputSidePacket(a, b) {
    Z(this, b, c => {
      var d = this.i._allocateDoubleVector(a.length);
      if (!d) throw Error('Unable to allocate new double vector on heap.');
      for (let e of a) this.i._addDoubleVectorEntry(d, e);
      this.i._addDoubleVectorToInputSidePacket(d, c)
    })
  }
  addFloatVectorToInputSidePacket(a, b) {
    Z(this, b, c => {
      var d = this.i._allocateFloatVector(a.length);
      if (!d) throw Error('Unable to allocate new float vector on heap.');
      for (let e of a) this.i._addFloatVectorEntry(d, e);
      this.i._addFloatVectorToInputSidePacket(d, c)
    })
  }
  addIntVectorToInputSidePacket(a, b) {
    Z(this, b, c => {
      var d = this.i._allocateIntVector(a.length);
      if (!d) throw Error('Unable to allocate new int vector on heap.');
      for (let e of a) this.i._addIntVectorEntry(d, e);
      this.i._addIntVectorToInputSidePacket(d, c)
    })
  }
  addUintVectorToInputSidePacket(a, b) {
    Z(this, b, c => {
      var d = this.i._allocateUintVector(a.length);
      if (!d)
        throw Error('Unable to allocate new unsigned int vector on heap.');
      for (let e of a) this.i._addUintVectorEntry(d, e);
      this.i._addUintVectorToInputSidePacket(d, c)
    })
  }
  addStringVectorToInputSidePacket(a, b) {
    Z(this, b, c => {
      var d = this.i._allocateStringVector(a.length);
      if (!d) throw Error('Unable to allocate new string vector on heap.');
      for (let e of a) Z(this, e, f => {this.i._addStringVectorEntry(d, f)});
      this.i._addStringVectorToInputSidePacket(d, c)
    })
  }
  attachBoolListener(a, b) {
    Vf(this, a, b);
    Z(this, a, c => {this.i._attachBoolListener(c)})
  }
  attachBoolVectorListener(a, b) {
    Wf(this, a, b);
    Z(this, a, c => {this.i._attachBoolVectorListener(c)})
  }
  attachIntListener(a, b) {
    Vf(this, a, b);
    Z(this, a, c => {this.i._attachIntListener(c)})
  }
  attachIntVectorListener(a, b) {
    Wf(this, a, b);
    Z(this, a, c => {this.i._attachIntVectorListener(c)})
  }
  attachUintListener(a, b) {
    Vf(this, a, b);
    Z(this, a, c => {this.i._attachUintListener(c)})
  }
  attachUintVectorListener(a, b) {
    Wf(this, a, b);
    Z(this, a, c => {this.i._attachUintVectorListener(c)})
  }
  attachDoubleListener(a, b) {
    Vf(this, a, b);
    Z(this, a, c => {this.i._attachDoubleListener(c)})
  }
  attachDoubleVectorListener(a, b) {
    Wf(this, a, b);
    Z(this, a, c => {this.i._attachDoubleVectorListener(c)})
  }
  attachFloatListener(a, b) {
    Vf(this, a, b);
    Z(this, a, c => {this.i._attachFloatListener(c)})
  }
  attachFloatVectorListener(a, b) {
    Wf(this, a, b);
    Z(this, a, c => {this.i._attachFloatVectorListener(c)})
  }
  attachStringListener(a, b) {
    Vf(this, a, b);
    Z(this, a, c => {this.i._attachStringListener(c)})
  }
  attachStringVectorListener(a, b) {
    Wf(this, a, b);
    Z(this, a, c => {this.i._attachStringVectorListener(c)})
  }
  attachProtoListener(a, b, c) {
    Vf(this, a, b);
    Z(this, a, d => {this.i._attachProtoListener(d, c || !1)})
  }
  attachProtoVectorListener(a, b, c) {
    Wf(this, a, b);
    Z(this, a, d => {this.i._attachProtoVectorListener(d, c || !1)})
  }
  attachAudioListener(a, b, c) {
    this.i._attachAudioListener ||
        console.warn(
            'Attempting to use attachAudioListener without support for output audio. Is build dep ":gl_graph_runner_audio_out" missing?');
    Vf(this, a, (d, e) => {
      d = new Float32Array(d.buffer, d.byteOffset, d.length / 4);
      b(d, e)
    });
    Z(this, a, d => {this.i._attachAudioListener(d, c || !1)})
  }
  finishProcessing() {
    this.i._waitUntilIdle()
  }
  closeGraph() {
    this.i._closeGraph();
    this.i.simpleListeners = void 0;
    this.i.emptyPacketListeners = void 0
  }
})),
    Yf = class extends Xf {};
async function Zf(a, b) {
  var c = await Tf(a.wasmLoaderPath, a.assetLoaderPath, {
    locateFile(d) {
      return d.endsWith('.wasm') ? a.wasmBinaryPath.toString() :
          a.assetBinaryPath && d.endsWith('.data') ?
                                   a.assetBinaryPath.toString() :
                                   d
    }
  });
  c.o.ja();
  c.oa = new Gf;
  await c.N(b);
  return c
}
async function $f(a, b) {
  return Zf(a, b)
}
function ag(a) {
  try {
    let b = a.H.length;
    if (b === 1) throw Error(a.H[0].message);
    if (b > 1)
      throw Error(
          'Encountered multiple errors: ' + a.H.map(c => c.message).join(', '));
  } finally {
    a.H = []
  }
}
function bg(a, b) {
  a.G = Math.max(a.G, b)
}
var cg = class {
  constructor(a) {
    this.o = a;
    this.H = [];
    this.G = 0;
    this.o.setAutoRenderToScreen(!1)
  }
  setGraph(a, b) {
    this.o.attachErrorListener((c, d) => {this.H.push(Error(d))});
    this.o.pa();
    this.o.setGraph(a, b);
    ag(this)
  }
  finishProcessing() {
    this.o.finishProcessing();
    ag(this)
  }
  close() {
    this.oa?.close();
    this.o.closeGraph()
  }
};
cg.prototype.close = cg.prototype.close;
export {cg as TaskRunner};
var dg = class extends J {
  constructor(a) {
    super(a)
  }
  j() {
    return Ob(B(this, 2)) ?? 0
  }
};
function eg(a, b) {
  Bc(a, 1, b)
}
function fg(a, b) {
  tc(a, 4, b)
}
var gg = class extends J {
  constructor(a) {
    super(a)
  }
};
var hg = [0, Zd, N, Qd, -1, M];
dg.prototype.l = $d(hg);
function ig(a, b, c) {
  a.h.push([b, c]);
  a.h.sort((d, e) => d[0] - e[0]);
  b = 0;
  for (let [d, e] of a.h) {
    c = d;
    let f = e;
    c <= b && (b = Math.max(b, c + f))
  }
  b === a.length && (a.data = void 0)
}
function jg(a, b, c, d) {
  if (a.data !== void 0) {
    var e = new Uint8Array(a.data.buffer, b, c);
    d === 1 && ig(a, b, c);
    return e
  }
}
class kg {
  constructor(a) {
    this.h = [];
    this.data = a;
    this.length = a.length
  }
}
function lg(a, b) {
  return new mg(async () => {
    var {value: c, done: d} = await a.read();
    return d ? void 0 : c
  }, b)
}
async function ng(a, b, c, d, e) {
  if (e === 2)
    return a.h = [], a.j = () => Promise.resolve(void 0),
           setTimeout(() => {a.l()}, 0), Promise.resolve(0);
  for (; a.size < c + d;) {
    var f = await a.j();
    if (f === void 0) break;
    a.h.push(new kg(f))
  }
  if (a.size < c + d)
    throw Error(
        `Data size is too small: ${a.size}, expected at least ${c + d}.`);
  f = b._malloc(d) >>> 0;
  var g = 0;
  for (let h = 0; h < a.h.length; h++) {
    let k = a.h[h];
    if (c >= k.length) {
      c -= k.length;
      continue
    }
    let l = Math.min(d, k.length - c);
    c = jg(k, c, l, e);
    if (c === void 0) throw Error('Data has already been released.');
    b.HEAPU8.set(c, f + g);
    c = 0;
    d -= l;
    g += l;
    if (d === 0) break
  }
  if (d !== 0) throw Error('Data not found.');
  return Promise.resolve(f)
}
var mg = class {
  constructor(a, b) {
    this.h = [];
    this.j = a;
    this.l = b
  }
  get size() {
    var a = 0;
    for (let b = 0; b < this.h.length; b++) a += this.h[b].length;
    return a
  }
};
function og(a) {
  return typeof a === 'object' && a != null && 'imageSource' in a
}
function pg(a) {
  return typeof a === 'object' && a != null && 'audioSource' in a
};
async function qg(a, b, c) {
  a = new rg(a, c);
  var d = 0;
  for (b = b.getReader();;) {
    let {value: e, done: f} = await b.read();
    if (f) break;
    a.set(e, d);
    d += e.byteLength
  }
  if (c !== d)
    throw sg(a),
        Error(
            `File could not be fully loaded to memory, so was not retained. Loaded ${
                d}/${c} bytes before failure`);
  return a
}
async function tg(a, b) {
  b = await fetch(b.toString());
  var c = Number(b.headers.get('content-length'));
  if (!b.body) throw Error('Response body is not available.');
  if (!c) throw Error('File size is 0.');
  return qg(a, b.body, c)
}
async function ug(a, b) {
  return qg(a, b.stream(), b.size)
}
function sg(a) {
  if (a.h) try {
      a.i._free(a.j)
    } catch {
    } finally {
      a.h = !1
    }
}
var rg = class {
  constructor(a, b) {
    this.i = a;
    this.l = b;
    this.j = this.i._malloc(b) >>> 0;
    this.m = this.i.HEAPU8;
    this.h = !!this.j
  }
  get offset() {
    if (!this.h) throw Error('WasmFileReference has been freed.');
    return this.j
  }
  get size() {
    if (!this.h) throw Error('WasmFileReference has been freed.');
    return this.l
  }
  set(a, b) {
    this.m.set(a, this.j + (b ?? 0))
  }
};
var vg = class extends J {
  constructor(a) {
    super(a)
  }
};
vg.prototype.j = $d([0, S, 2, R, N, P]);
var wg = class extends J {
  constructor(a) {
    super(a)
  }
};
var xg = class extends J {
  constructor(a) {
    super(a)
  }
};
var yg = class extends J {
  constructor(a) {
    super(a)
  }
};
var zg = class extends J {
  constructor(a) {
    super(a)
  }
};
var Ag = [
  0,
  N,
  -6,
  1,
  N,
  1,
  [0, P, Zd, -2],
  [0, P, Qd],
  Zd,
  -2,
  [0, P, -1, Zd, Qd, W, M, O, -2],
  1,
  P,
  N,
  M,
  -1,
  [0, Zd, N],
  P,
  -1,
  Pd,
  N,
  -5,
  Pd,
  -1,
  [0, M, Pd],
  M,
  O,
  [0, M, -2],
  Pd,
  [0, N],
  [0, N, -4],
  O,
  M,
  -2,
  O,
  -1,
  Qd,
  Pd,
  O,
  N,
  -1,
  [0, M, -2],
  P,
  N,
  M,
  O,
  [0, M, -1],
  M,
  O,
  -1
];
var Bg = [0, S, -2];
var Cg = [
  0,
  [4, 6],
  Ag,
  N,
  1,
  Ud,
  R,
  Vd,
  Yd,
  Bg,
  M,
  [
    0, [0, N, -1, U, [0, N, [0, N, -1], -1, [0, Zd, -1], P], P, -2, N, -1],
    [0, N, -1, P], Ag, P, N, [0, N], -1
  ],
  Q,
  -3,
  [0, N, P],
  Ag,
  [0, Bg, -2],
  Td,
  U,
  [0, S, -2],
  Td
];
zg.prototype.j = $d([
  0,
  S,
  8,
  [0, P, -6],
  1,
  N,
  1,
  N,
  [0, U, [0, S, Sd, -1, Zd], Cg, N],
  [0, N, P, -3],
  1,
  Zd,
  1,
  Cg,
  1,
  N,
  5,
  Zd,
  Td,
  1,
  hg,
  P,
  N,
  P,
  -1
]);
var Dg = class extends J {
  constructor(a) {
    super(a)
  }
};
var Eg = class extends J {
  constructor(a) {
    super(a)
  }
}
, xc = [2, 4];
Eg.prototype.j = $d([0, xc, N, Vd, N, V, [0, 1, S]]);
var Fg = function(a) {
  return class extends a {
    constructor() {
      super(...arguments);
      this.P = !1;
      this.J = this.K = 0
    }
    M() {
      if (this.P)
        throw Error(
            'Cannot process because LLM inference engine is currently loading or processing.');
      this.P = !0
    }
    L() {
      this.P = !1
    }
    async createLlmInferenceEngine(b, c) {
      this.M();
      try {
        let d = lg(b, () => {});
        await this.i.createLlmInferenceEngine(
            D(c, 2) ?? 512, zc(c, dg, 3)?.j() ?? 40, A(B(c, 6)) ?? !1 ?? !1,
            D(c, 7) ?? 0, A(B(c, 8)) ?? !1 ?? !1,
            (e, f, g) => ng(d, this.i, e, f, g))
      } finally {
        this.L()
      }
    }
    async ea(b, c) {
      this.M();
      try {
        await this.ra(b),
            await this.i.ccall(
                'CreateLlmInferenceEngineConverted', 'void',
                ['number', 'number', 'boolean'],
                [
                  D(c, 2) ?? 512, zc(c, dg, 3)?.j() ?? 40,
                  A(B(c, 6)) ?? !1 ?? !1
                ],
                {async: !0})
      } finally {
        this.L()
      }
    }
    W() {
      this.M();
      try {
        let b = this.i;
        b.ccall('DeleteLlmInferenceEngine', 'void', [], [], {async: !1});
        this.K &&
            (b._FreeSession(this.K), this.J === this.K && (this.J = 0),
             this.K = 0);
        this.J && (b._FreeSession(this.J), this.J = 0)
      } finally {
        this.L()
      }
    }
    async R(b, c, d) {
      this.M();
      try {
        let e = [], f = this.i;
        f._userProgressListener = (u, z) => {
          u && e.push(u);
          d && d(u, z)
        };
        let g = c.l(), h = g.length, k = this.i._malloc(h);
        this.i.HEAPU8.set(g, k);
        let l = b.some(pg), p = b.some(og);
        f.ccallNum = f.ccall;
        let t = await f.ccallNum(
            'MakeSessionForPredict', 'number',
            ['number', 'number', 'boolean', 'boolean'], [k, h, p, l],
            {async: !0});
        c = [];
        for (let u of b)
          if (typeof u === 'string')
            Z(this, u, z => {f._AddTextQueryChunk(t, z)});
          else if (og(u)) {
            let {image: z, width: T, height: xa} = await this.ia(u.imageSource),
                                             fa = typeof OffscreenCanvas !==
                    'undefined' ?
                new OffscreenCanvas(T, xa) :
                document.createElement('canvas');
            fa.width = T;
            fa.height = xa;
            let ya = fa.getContext('2d');
            ya.drawImage(z, 0, 0);
            let Y = ya.getImageData(0, 0, T, xa),
                za = this.i._malloc(Y.width * Y.height * 4);
            this.i.HEAPU8.set(Y.data, za);
            f._AddImageQueryChunk(t, za, Y.width, Y.height);
            c.push(za)
          } else if (pg(u)) {
            let z = await this.ha(u.audioSource),
                T = this.i._malloc(z.audioSamples.byteLength);
            this.i.HEAPF32.set(z.audioSamples, T / 4);
            f._AddAudioQueryChunk(
                t, z.audioSampleRateHz, T, z.audioSamples.length);
            c.push(T)
          } else
            throw Error('Unsupported PromptPart type in query.');
        await f.ccall('PredictSession', 'void', ['number'], [t], {async: !0});
        b = !0;
        p && this.J === 0 && (this.J = t, b = !1);
        l && this.K === 0 && (this.K = t, b = !1);
        b && f._FreeSession(t);
        for (let u of c) this.i._free(u);
        c.length = 0;
        d && d('', !0);
        this.i._free(k);
        f._userProgressListener = void 0;
        return e.join('')
      } finally {
        this.L()
      }
    }
    S(b) {
      this.M();
      var c = 0, d = '';
      for (let e of b)
        typeof e === 'string' ? d += e :
            og(e)             ? c += 260 :
                                pg(e) &&
                console.warn(
                    'sizeInTokens is not yet implemented for audio; audio tokens will not be counted');
      try {
        let e;
        Z(this, d, f => {e = this.i._GetSizeInTokens(f)});
        return c + e
      } finally {
        this.L()
      }
    }
    async ra(b) {
      b = await Jf(b);
      try {
        this.i.FS_unlink('llm.task')
      } catch {
      }
      this.i.FS_createDataFile('/', 'llm.task', b, !0, !1, !1)
    }
    async ia(b) {
      if (typeof b === 'string') {
        let c = new Image;
        c.src = b;
        c.crossOrigin = 'Anonymous';
        try {
          await c.decode()
        } catch {
          throw Error(`Image from URL ${b} failed to load`);
        }
        return {
          image: c, width: c.naturalWidth, height: c.naturalHeight
        }
      }
      if (b instanceof HTMLImageElement) {
        try {
          await b.decode()
        } catch {
          throw Error('Image from HTMLImageElement failed to load');
        }
        return {
          image: b, width: b.naturalWidth, height: b.naturalHeight
        }
      }
      return b instanceof HTMLVideoElement ?
          {image: b, width: b.videoWidth, height: b.videoHeight} :
          b instanceof VideoFrame ?
          {image: b, width: b.displayWidth, height: b.displayHeight} :
          {image: b, width: b.width, height: b.height}
    }
    async ha(b) {
      if (typeof b === 'string') {
        let c = await fetch(b);
        if (!c.ok) throw Error(`Audio fetch for ${b} had error: ${c.status}`);
        b = await c.arrayBuffer();
        b = await (new AudioContext({sampleRate: 16E3})).decodeAudioData(b);
        return {
          audioSamples: b.getChannelData(0), audioSampleRateHz: b.sampleRate
        }
      }
      return typeof b === 'object' && b != null && 'audioSamples' in b &&
              'audioSampleRateHz' in b ?
          b :
          {audioSamples: b.getChannelData(0), audioSampleRateHz: b.sampleRate}
    }
  }
}(function(a) {
  var b = class extends a {
    static async qa(c, d) {
      d ||= await b.Y();
      var e = [];
      for (let g of c?.requiredFeatures ?? [])
        d.features.has(g) ?
            e.push(g) :
            console.warn(`WebGPU feature ${g} is not supported.`);
      c = {...c, requiredFeatures: e};
      try {
        var f = await d.requestDevice(c)
      } catch (g) {
        throw console.error(
            'Unable to initialize WebGPU with the requested features.'),
            g;
      }
      c = f;
      c.adapterInfo || (c.adapterInfo = d.info);
      return f
    }
    static async Y(c) {
      c = await navigator.gpu.requestAdapter(c);
      if (!c)
        throw Error(
            'Unable to request adapter from navigator.gpu; Ensure WebGPU is enabled.');
      return c
    }
    ka(c) {
      if (d)
        typeof HTMLCanvasElement !== 'undefined' &&
            d instanceof HTMLCanvasElement && (d.id = 'canvas_webgpu');
      else
        var d = new OffscreenCanvas(1, 1);
      d.getContext('webgpu').configure(
          {device: c, format: navigator.gpu.getPreferredCanvasFormat()});
      this.i.preinitializedWebGPUDevice = c
    }
    da() {
      return this.i.ccall('closeGraph', 'void', [], [], {async: !0})
    }
  };
  return b
}(function(a) {
  return class extends a {
    addStreamingReaderToInputSidePacket(b, c) {
      this.i.addStreamingReaderToInputSidePacket(
          (d, e, f) => ng(b, this.i, d, e, f), c)
    }
  }
}(function(a) {
  return class extends a {
    ba(b, c) {
      Z(this, 'lora_model_ref_in',
        d => {this.i._addRawDataSpanToInputStream(b.offset, b.size, d, c)})
    }
  }
}(Yf))));
class Gg extends Fg {}
var Ig = class {
  constructor(a) {
    this.j = a;
    this.h = Hg;
    Hg++
  }
}
, Hg = 1;
class Jg {
  constructor() {
    var a, b;
    this.promise = new Promise((c, d) => {
      a = c;
      b = d
    });
    this.resolve = a;
    this.reject = b
  }
}
function Kg(a) {
  return a === 1 ? 1 : a + a % 2
}
async function Lg() {
  var a = await Gg.Y({powerPreference: 'high-performance'}),
      b = a.limits.maxBufferSize, c = a.limits.maxStorageBufferBindingSize;
  b < 524550144 &&
      console.warn(
          `This WebGPU device is unable to execute most LLM tasks, because the required maxBufferSize is usually at least 524550144, but your device only supports maxBufferSize of ${
              b}`);
  c < 524550144 &&
      console.warn(
          `The WebGPU device is unable to execute LLM tasks, because the required maxStorageBufferBindingSize is usually at least 524550144, but your device only supports maxStorageBufferBindingSize of ${
              c}`);
  b = {
    requiredFeatures: ['shader-f16'],
    requiredLimits: {
      maxStorageBufferBindingSize: c,
      maxBufferSize: b,
      maxStorageBuffersPerShaderStage: a.limits.maxStorageBuffersPerShaderStage
    }
  };
  a.features.has('subgroups') &&
      (console.warn(
           'Experimental Chromium WGSL subgroup support detected. Enabling this feature in the inference engine.'),
       b.requiredFeatures = ['shader-f16', 'subgroups']);
  return Gg.qa(b, a)
}
function Mg(a, b) {
  return a == null || a.length === 0 ? [] : a.map(c => {
    c = c.replaceAll('\u2581', ' ');
    c = c.replaceAll('<0x0A>', '\n');
    b && (c = c.trimStart());
    return c.split('\\[eod\\]', 1)[0]
  })
}
function Ng(a) {
  if (a.C.length > 0) {
    let b = [...a.C];
    a.C.length = 0;
    if (a.l)
      a.l.reject(b), a.l = void 0;
    else
      throw b;
  }
}
function Og(a) {
  var b = Pg(a);
  a.o.attachStringVectorListener('text_out', (d, e) => {
    d = Mg(d, a.F.length === 0);
    d.forEach((f, g) => {g < D(a.h, 5) && a.F[g].push(f)});
    a.A && a.C.length === 0 &&
        (a.B ? (d.length > D(a.h, 5) && d.pop(), a.A(d, !1)) : a.A(d[0], !1));
    bg(a, e)
  });
  a.o.attachEmptyPacketListener('text_out', d => {bg(a, d)});
  a.o.attachBoolListener('text_end', (d, e) => {
    bg(a, e);
    try {
      Ng(a)
    } catch (f) {
      throw a.j = !1, f;
    }
    a.l && (a.l.resolve(a.F.map(f => f.join(''))), a.l = void 0);
    if (a.A)
      if (a.B) {
        d = [];
        for (e = 0; e < D(a.h, 5); e++) d.push('');
        a.A(d, !0)
      } else
        a.A('', !0);
    a.j = !1;
    a.B = void 0
  });
  a.o.attachEmptyPacketListener('text_end', d => {
    a.j = !1;
    a.B = void 0;
    bg(a, d);
    Ng(a);
    a.l && (a.l.resolve(a.F.map(e => e.join(''))), a.l = void 0)
  });
  a.o.attachIntListener('token_cost_out', (d, e) => {
    a.U = d;
    bg(a, e)
  });
  a.V && a.o.addStreamingReaderToInputSidePacket(a.V, 'streaming_reader');
  var c = b.j();
  a.D?.removeEventListener('uncapturederror', a.I);
  return a.o.da().then(() => {
    a.D?.addEventListener('uncapturederror', a.I);
    a.C.length = 0;
    a.setGraph(new Uint8Array(c), !0);
    a.finishProcessing()
  })
}
function Qg(a, b, c, d) {
  a.A = typeof c === 'function' ? c : d;
  b = Array.isArray(b) ? b : [b];
  d = b.filter(e => og(e)).length;
  if (d > 0 && (Pb(B(a.h, 7)) == null || D(a.h, 7) < d))
    throw Error(`maxNumImages is set to ${
        Pb(B(a.h, 7)) != null ? D(a.h, 7) :
                                0}, but the query included ${d} images.`);
  d = b.filter(e => pg(e)).length;
  if (d > 0 && (A(B(a.h, 8)) == null || !(A(B(a.h, 8)) ?? !1)))
    throw Error(`supportAudio was not enabled, but the query included ${
        d} audio chunks.`);
  if (a.u) {
    if (a.B && D(a.h, 5) > 1)
      throw Error(
          'Multi-response generation is not supported for converted LLM models (.task format) yet, nor is it supported for multimodality. Please use the .bin format without multimodality or request only one response.');
    if (c instanceof Ig)
      throw Error(
          'LoRA is not supported for converted LLM models (.task format) yet, nor is it supported for multimodality. Please use the .bin format without multimodality to use LoRA.');
    a.o.i.LLM_CANCEL_FLAG = void 0;
    return a.o
        .R(b, a.m,
           (e,
            f) => {a.C.length === 0 && a.A && (a.B ? a.A([e], f) : a.A(e, f))})
        .then(e => {
          Ng(a);
          return [e]
        })
  }
  if (a.j) throw Error('Previous invocation or loading is still ongoing.');
  a.j = !0;
  a.o.i.LLM_CANCEL_FLAG = void 0;
  a.F.length = 0;
  for (d = 0; d < D(a.h, 5); d++) a.F[d] = [];
  d = a.G + 1;
  a.o.addStringToStream(b.join(''), 'text_in', d);
  if (c instanceof Ig) {
    if (c.j !== a)
      throw a.j = !1, a.B = void 0,
            Error('The LoRA model was not loaded by this LLM Inference task.');
    a.o.addUintToStream(c.h, 'lora_model_id_to_apply_in', d)
  } else
    a.o.addEmptyPacketToStream('lora_model_id_to_apply_in', d);
  a.finishProcessing();
  a.l = new Jg;
  return a.l.promise
}
function Pg(a) {
  var b = new zf;
  G(b, 10, 'text_in');
  G(b, 10, 'token_cost_in');
  G(b, 10, 'lora_model_id_to_apply_in');
  G(b, 10, 'lora_model_ref_in');
  G(b, 10, 'lora_model_id_to_load_in');
  G(b, 16, 'streaming_reader');
  G(b, 15, 'text_out');
  G(b, 15, 'text_end');
  G(b, 15, 'token_cost_out');
  var c = new uf;
  F(c, 2, 'TokenizerInputBuildCalculator');
  G(c, 3, 'PROMPT:text_in');
  G(c, 3, 'LORA_ID:lora_model_id_to_apply_in');
  G(c, 4, 'prompt');
  yf(b, c);
  c = new uf;
  F(c, 2, 'ModelDataCalculator');
  G(c, 6, 'MODEL_DATA:__side_packet_1');
  G(c, 6, 'MODEL_TYPE:model_type');
  G(c, 5, 'READ_DATA_FN:streaming_reader');
  G(c, 3, 'LORA_MODEL_SPAN:lora_model_ref_in');
  G(c, 3, 'LORA_MODEL_ID:lora_model_id_to_load_in');
  G(c, 4, 'LORA_DATA:lora_model_data');
  yf(b, c);
  c = new uf;
  F(c, 2, 'Gpt2UnicodeMappingCalculator');
  G(c, 5, 'MODEL_TYPE:model_type');
  G(c, 6, 'BYTES_TO_UNICODE_MAPPING:tokenizer_mapping');
  yf(b, c);
  c = new be;
  F(c, 1, 'type.googleapis.com/odml.infra.proto.TokenizerCalculatorOptions');
  var d = new Eg, e = D(a.h, 2);
  E(d, 1, e);
  e = new Dg;
  F(e, 2, 'spm_vocab_model');
  e = Ac(e);
  a: {
    jc(d); var f = d.v, g = f[r] | 0; if (e == null) {
      var h = vc(f);
      if (wc(h, f, g) === 4)
        h.set(xc, 0);
      else
        break a
    } else {
      h = f;
      let k = vc(h), l = wc(k, h, g);
      l !== 4 && (l && (g = C(h, g, l)), k.set(xc, 4))
    } C(f, g, 4, e)
  } e &&
      !w(e) && kc(d.v);
  E(d, 3, 2);
  ae(c, d.j());
  d = new uf;
  F(d, 2, 'TokenizerCalculator');
  Cc(d, 8, be, c);
  G(d, 5, 'MODEL_DATA:__side_packet_1');
  G(d, 3, 'PROMPT_AND_INPUT_OPTIONS:prompt');
  G(d, 5, 'BYTES_TO_UNICODE_MAPPING:tokenizer_mapping');
  G(d, 6, 'PROCESSOR_GETTER:__input_side_1');
  G(d, 4, 'IDS_AND_INPUT_OPTIONS:__stream_0');
  yf(b, d);
  c = new be;
  F(c, 1, 'type.googleapis.com/odml.infra.proto.LlmGpuCalculatorOptions');
  d = new zg;
  E(d, 12, 3);
  F(d, 1, 'llm.tflite');
  E(d, 14, 0);
  e = Kg(D(a.h, 5));
  E(d, 22, e);
  e = zc(a.h, dg, 3);
  Bc(d, 31, e);
  e = new wg;
  Ec(e, 1, !0);
  A(B(a.h, 6)) != null && (A(B(a.h, 6)) ?? !1) && Ec(e, 1, !1);
  Ec(e, 2, !0);
  Ec(e, 5, !0);
  Bc(d, 10, e);
  e = nc(a.h, 4, Ob, void 0 === fb ? 2 : 4);
  tc(d, 29, e);
  e = new yg;
  f = new xg;
  E(f, 1, 1);
  h = D(a.h, 2);
  E(f, 2, h);
  A(B(a.h, 9)) != null && (A(B(a.h, 9)) ?? !1) &&
      (Ec(d, 35, !0), Dc(f, 60, !0));
  Bc(e, 1, f);
  Bc(d, 20, e);
  ae(c, d.j());
  d = new uf;
  F(d, 2, 'LlmGpuCalculator');
  Cc(d, 8, be, c);
  G(d, 3, 'IDS_AND_INPUT_OPTIONS:__stream_0');
  G(d, 3, 'FINISH:finish');
  G(d, 3, 'LORA_DATA:lora_model_data');
  G(d, 5, 'MODEL_DATA:__side_packet_1');
  G(d, 4, 'DECODED_IDS:__stream_3');
  G(d, 4, 'OUTPUT_END:__stream_4');
  c = new sf;
  F(c, 1, 'FINISH');
  Ec(c, 2, !0);
  Cc(d, 13, sf, c);
  yf(b, d);
  c = new uf;
  F(c, 2, 'IsPacketPresentCalculator');
  G(c, 3, '__stream_4');
  G(c, 4, 'text_end');
  yf(b, c);
  c = new be;
  F(c, 1, 'type.googleapis.com/odml.infra.proto.DetokenizerCalculatorOptions');
  d = new vg;
  a = Kg(D(a.h, 5));
  E(d, 5, a);
  G(d, 4, '<eos>');
  G(d, 4, '<|endoftext|>');
  ae(c, d.j());
  a = new uf;
  F(a, 2, 'DetokenizerCalculator');
  Cc(a, 8, be, c);
  G(a, 3, 'IDS_AND_INPUT_OPTIONS:__stream_3');
  G(a, 5, 'PROCESSOR_GETTER:__input_side_1');
  G(a, 5, 'BYTES_TO_UNICODE_MAPPING:tokenizer_mapping');
  G(a, 5, 'MODEL_DATA:__side_packet_1');
  G(a, 4, 'FINISH_AND_INPUT_OPTIONS:finish');
  G(a, 4, 'WORDS:text_out');
  yf(b, a);
  a = new uf;
  F(a, 2, 'TokenCostCalculator');
  G(a, 3, 'PROMPT:token_cost_in');
  G(a, 5, 'PROCESSOR_GETTER:__input_side_1');
  G(a, 5, 'BYTES_TO_UNICODE_MAPPING:tokenizer_mapping');
  G(a, 4, 'NUM_TOKENS:token_cost_out');
  yf(b, a);
  return b
}
var X = class extends cg {
  constructor(a, b) {
    super(new Gg(a, b));
    this.F = [];
    this.O = this.u = this.j = !1;
    this.C = [];
    this.I = c => {
      c = c.error;
      if (c.message.match(/exceeds the max buffer size limit/))
        throw Error(
            'Failed to run this LLM model because it requires a buffer size that exceeds the maximum size your device supports, but you could try a smaller LLM model or different device.\nWebGPU throws: ' +
            `"${c.message}"`);
      if (c.message.match(
              /is larger than the maximum storage buffer binding size/))
        throw Error(
            'Failed to run this LLM model because it requires a storage buffer binding size that exceeds the maximum size your device supports, but you could try a smaller LLM model or different device.\n' +
            `WebGPU throws: "${c.message}"`);
      this.C.push(c)
    };
    this.h = new gg;
    eg(this.h, new Af);
    this.m = new dg;
    Bc(this.h, 3, this.m);
    Fc(this.h, 2, 512);
    a = this.m;
    if (!Fb(2)) throw Pa('enum');
    uc(a, 1, 2, 0);
    E(this.m, 2, 40);
    uc(this.m, 3, Hb(1), 0);
    mc(this.m, 5, Nb(0));
    uc(this.m, 4, Hb(.8), 0);
    Fc(this.h, 5, 1)
  }
  async N(a) {
    if (this.j) throw Error('Cannot set options while loading or processing.');
    if (a.baseOptions?.modelAssetPath && a.baseOptions?.modelAssetBuffer)
      throw Error(
          'Cannot set both baseOptions.modelAssetPath and baseOptions.modelAssetBuffer');
    var b, c = new Promise(g => {b = g});
    if (a.baseOptions?.modelAssetPath) {
      var d = await fetch(a.baseOptions.modelAssetPath.toString());
      if (!d.ok)
        throw Error(`Failed to fetch model: ${a.baseOptions.modelAssetPath} (${
            d.status})`);
      if (!d.body)
        throw Error(
            `Failed to fetch model: ${a.baseOptions.modelAssetPath} (no body)`);
      d = d.body.getReader()
    } else
      a.baseOptions?.modelAssetBuffer instanceof Uint8Array ?
          d = Pf(a.baseOptions.modelAssetBuffer).getReader() :
          a.baseOptions?.modelAssetBuffer instanceof
              ReadableStreamDefaultReader ?
          (d = a.baseOptions.modelAssetBuffer,
           a.baseOptions.modelAssetBuffer = void 0) :
          b();
    if (d) {
      let [g, h] = Hf(d);
      var e = await Of(h);
      this.O = e === 1;
      var f = null;
      if (e === 2) {
        let l = this.o.i;
        f = await Nf(g, p => {
          var t = l._malloc(p.length);
          l.HEAPU8.set(p, t);
          p = l._GetLiteRtModelOffset(t);
          l._free(t);
          return p
        })
      }
      e = 'maxNumImages' in a && a.maxNumImages ? a.maxNumImages : 0;
      Fc(this.h, 7, e);
      let k = 'supportAudio' in a && !!a.supportAudio;
      Dc(this.h, 8, k);
      this.O || e > 0 || k ? (this.u = !0, d = f ? f : g) :
                             (this.u = !1, this.V = lg(f ? f : g, b))
    } else
      throw Error('No model asset provided.');
    a.baseOptions?.gpuOptions?.device &&
        (this.D && this.D.removeEventListener('uncapturederror', this.I),
         this.D = a.baseOptions.gpuOptions.device, this.o.ka(this.D),
         this.D.addEventListener('uncapturederror', this.I));
    'maxTokens' in a && Fc(this.h, 2, a.maxTokens ?? 512);
    'topK' in a && E(this.m, 2, a.topK ?? 40);
    'temperature' in a && uc(this.m, 4, Hb(a.temperature ?? .8), 0);
    'randomSeed' in a && mc(this.m, 5, Nb(a.randomSeed ?? 0));
    'loraRanks' in a && fg(this.h, a.loraRanks ?? []);
    if ('numResponses' in a) {
      f = a.numResponses ?? 1;
      if (f < 1) throw Error('\'numResponses\' must be at least 1.');
      if (this.u && f > 1)
        throw Error(
            '\'numResponses > 1\' is not supported for converted LLM models yet, and is also not supported with multimodality.');
      Fc(this.h, 5, f);
      e = zc(this.h, dg, 3);
      f > 1 && e && (e.j() <= 1 || (B(e, 4, Ib) ?? 0) <= 0) &&
          console.warn(
              'To generate multiple responses, it is expected topK > 1 and temperature > 0; otherwise, all the generated responses may be the same.')
    }
    'forceF32' in a && a.forceF32 !== void 0 && Dc(this.h, 6, a.forceF32);
    if ('disableRewinding' in a && a.disableRewinding !== void 0) {
      if (this.u && a.disableRewinding)
        throw Error(
            '\'disableRewinding\' is not supported for converted LLM models yet, and is also not supported with multimodality.');
      Dc(this.h, 9, a.disableRewinding)
    }
    if (this.u)
      return this.o.W(),
             this.O ? this.o.ea(d, this.h).then(() => {Ng(this)}) :
                      this.o.createLlmInferenceEngine(d, this.h).then(
                          () => {Ng(this)});
    this.j = !0;
    a = Og(this).then(() => {});
    return Promise.all([c, a]).then(() => {
      this.j = !1;
      Ng(this)
    })
  }
  get baseOptions() {
    return zc(this.h, Af, 1)
  }
  set baseOptions(a) {
    eg(this.h, a)
  }
  get isIdle() {
    return !this.j && !this.l
  }
  R(a, b, c) {
    D(this.h, 5) > 1 &&
        console.warn(
            '\'numResponses\' is set larger than 1 and this function only returns the first response, so we recommend either using \'generateResponses()\' to obtain multiple responses, or else setting \'numResponses\' to 1 for better performance.');
    this.B = !1;
    return Qg(this, a, b, c).then(d => d[0])
  }
  ga(a, b, c) {
    this.B = !0;
    return Qg(this, a, b, c)
  }
  S(a) {
    a = Array.isArray(a) ? a : [a];
    if (this.u) return this.o.S(a);
    if (this.j) throw Error('Previous invocation or loading is still ongoing.');
    if (a.some(og))
      throw Error('sizeInTokens requires maxNumImages > 0 for images.');
    if (a.some(pg))
      throw Error('sizeInTokens requires supportAudio for audio.');
    a = a.join('');
    this.j = !0;
    this.U = void 0;
    this.o.addStringToStream(a, 'token_cost_in', this.G + 1);
    this.finishProcessing();
    this.j = !1;
    return this.U
  }
  ca() {
    var a = this.o.i;
    if (this.u || this.j) a.LLM_CANCEL_FLAG = 1
  }
  async na(a) {
    if (this.u)
      throw Error(
          'LoRA is not supported for converted LLM models (.task format) yet, nor is it supported for multimodality. Please use the old format (.bin) without multimodality to use LoRA.');
    if (this.j)
      throw Error('Cannot load LoRA model while loading or processing.');
    this.j = !0;
    if (a instanceof Uint8Array) {
      var b = new rg(this.o.i, a.length);
      b.set(a);
      a = b
    } else
      a = a instanceof Blob ? await ug(this.o.i, a) : await tg(this.o.i, a);
    b = new Ig(this);
    var c = this.G + 1;
    this.o.ba(a, c);
    this.o.addUintToStream(b.h, 'lora_model_id_to_load_in', c);
    this.finishProcessing();
    sg(a);
    bg(this, c);
    this.j = !1;
    return b
  }
  close() {
    this.u && this.o.W();
    this.D?.removeEventListener('uncapturederror', this.I);
    super.close()
  }
};
X.prototype.loadLoraModel = X.prototype.na;
X.prototype.cancelProcessing = X.prototype.ca;
X.prototype.sizeInTokens = X.prototype.S;
X.prototype.generateResponses = X.prototype.ga;
X.prototype.generateResponse = X.prototype.R;
X.prototype.setOptions = X.prototype.N;
X.createWebGpuDevice = Lg;
X.createFromModelPath = async function(a, b) {
  b = {baseOptions: {gpuOptions: {device: await Lg()}, modelAssetPath: b}};
  return $f(a, b)
};
X.createFromModelBuffer = async function(a, b) {
  b = {baseOptions: {gpuOptions: {device: await Lg()}, modelAssetBuffer: b}};
  return $f(a, b)
};
X.createFromOptions = async function(a, b) {
  if (!b.baseOptions?.gpuOptions?.device) {
    let c = await Lg();
    b.baseOptions = b.baseOptions ?? {};
    b.baseOptions.gpuOptions = b?.baseOptions?.gpuOptions ?? {};
    b.baseOptions.gpuOptions.device = c
  }
  return $f(a, b)
};
export {X as LlmInference};
