// frontend_assets.h — eingebettete Frontends (Design §18 static/).
// index.html (Desktop) + mobile.html als raw string literals — Teil von nova4.exe.
#pragma once

namespace nova::web {
const char* index_html();   // Desktop
const char* mobile_html();  // Mobile
}  // namespace nova::web
