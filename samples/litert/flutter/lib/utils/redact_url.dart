// Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/// [url] without the parts that may carry a secret, for logs, the screen and
/// the diagnostics report: no user-info (`user:token@`), no query (a signed
/// link's token) and no fragment. Scheme, host, port and path stay, so the
/// line still says where it points.
///
/// `https://user:token@host:8443/a/model.litertlm?sig=x#f` becomes
/// `https://host:8443/a/model.litertlm`.
String redactUrl(Uri url) => Uri(
  scheme: url.scheme,
  host: url.hasAuthority ? url.host : null,
  port: url.hasPort ? url.port : null,
  path: url.path,
).toString();
