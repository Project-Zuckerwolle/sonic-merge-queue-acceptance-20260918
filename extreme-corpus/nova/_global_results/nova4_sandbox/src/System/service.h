// service.h — Windows Service / SCM (Design §1b, §18).
//
// [Server-PC] Nova läuft als Windows Service (Autostart nach Boot, ohne Login).
//   nova4.exe --install-service   -> sc create + Autostart
//   nova4.exe --uninstall-service -> Service entfernen
// CUDA Compute aus Session 0 funktioniert für ML-Workloads (§1b).
#pragma once

#include <functional>
#include <string>

namespace nova::system {

enum class ServiceCommand { None, Install, Uninstall, Run };

// Parst die Kommandozeile (testbar, ohne Windows-API).
ServiceCommand parse_service_command(int argc, char** argv);

constexpr const char* kServiceName = "Nova4";
constexpr const char* kServiceDisplay = "Nova 4 Local Assistant";

#ifdef _WIN32
// Registriert/entfernt den Dienst (benötigt Admin-Rechte). Liefert false + err.
bool install_service(std::string* err = nullptr);
bool uninstall_service(std::string* err = nullptr);

// Startet die SCM-Dispatcher-Schleife; main_body läuft als Service-Worker und
// soll laufen bis stop_requested() true wird.
bool run_as_service(std::function<void()> main_body, std::string* err = nullptr);
bool service_stop_requested();
#endif

}  // namespace nova::system
