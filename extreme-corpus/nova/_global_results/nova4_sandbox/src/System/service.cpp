// service.cpp — Implementierung von service.h (Windows SCM).
#include "System/service.h"

#include <cstring>

#ifdef _WIN32
#include <windows.h>
#include <atomic>
#endif

namespace nova::system {

ServiceCommand parse_service_command(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--install-service") == 0)   return ServiceCommand::Install;
        if (std::strcmp(argv[i], "--uninstall-service") == 0) return ServiceCommand::Uninstall;
        if (std::strcmp(argv[i], "--service") == 0)           return ServiceCommand::Run;
    }
    return ServiceCommand::None;
}

#ifdef _WIN32
namespace {
SERVICE_STATUS        g_status{};
SERVICE_STATUS_HANDLE g_status_handle = nullptr;
std::atomic<bool>     g_stop{false};
std::function<void()> g_body;

void set_state(DWORD state) {
    g_status.dwCurrentState = state;
    g_status.dwControlsAccepted = (state == SERVICE_RUNNING) ? SERVICE_ACCEPT_STOP : 0;
    if (g_status_handle) ::SetServiceStatus(g_status_handle, &g_status);
}

void WINAPI service_ctrl(DWORD ctrl) {
    if (ctrl == SERVICE_CONTROL_STOP || ctrl == SERVICE_CONTROL_SHUTDOWN) {
        g_stop.store(true);
        set_state(SERVICE_STOP_PENDING);
    }
}

void WINAPI service_main(DWORD, LPSTR*) {
    g_status_handle = ::RegisterServiceCtrlHandlerA(kServiceName, service_ctrl);
    if (!g_status_handle) return;
    g_status.dwServiceType = SERVICE_WIN32_OWN_PROCESS;
    set_state(SERVICE_START_PENDING);
    set_state(SERVICE_RUNNING);
    if (g_body) g_body();
    set_state(SERVICE_STOPPED);
}
}  // namespace

bool service_stop_requested() { return g_stop.load(); }

bool install_service(std::string* err) {
    char path[MAX_PATH];
    if (!::GetModuleFileNameA(nullptr, path, MAX_PATH)) { if (err) *err = "GetModuleFileName"; return false; }
    std::string bin = std::string("\"") + path + "\" --service";

    SC_HANDLE scm = ::OpenSCManagerA(nullptr, nullptr, SC_MANAGER_CREATE_SERVICE);
    if (!scm) { if (err) *err = "OpenSCManager (Admin nötig)"; return false; }
    SC_HANDLE svc = ::CreateServiceA(scm, kServiceName, kServiceDisplay,
        SERVICE_ALL_ACCESS, SERVICE_WIN32_OWN_PROCESS, SERVICE_AUTO_START,
        SERVICE_ERROR_NORMAL, bin.c_str(), nullptr, nullptr, nullptr, nullptr, nullptr);
    bool ok = svc != nullptr;
    if (!ok && err) *err = "CreateService fehlgeschlagen (existiert evtl. schon)";
    if (svc) ::CloseServiceHandle(svc);
    ::CloseServiceHandle(scm);
    return ok;
}

bool uninstall_service(std::string* err) {
    SC_HANDLE scm = ::OpenSCManagerA(nullptr, nullptr, SC_MANAGER_ALL_ACCESS);
    if (!scm) { if (err) *err = "OpenSCManager (Admin nötig)"; return false; }
    SC_HANDLE svc = ::OpenServiceA(scm, kServiceName, DELETE | SERVICE_STOP);
    if (!svc) { if (err) *err = "OpenService (nicht installiert?)"; ::CloseServiceHandle(scm); return false; }
    SERVICE_STATUS st{};
    ::ControlService(svc, SERVICE_CONTROL_STOP, &st);
    const bool ok = ::DeleteService(svc) != FALSE;
    if (!ok && err) *err = "DeleteService fehlgeschlagen";
    ::CloseServiceHandle(svc);
    ::CloseServiceHandle(scm);
    return ok;
}

bool run_as_service(std::function<void()> main_body, std::string* err) {
    g_body = std::move(main_body);
    SERVICE_TABLE_ENTRYA table[] = {
        {const_cast<char*>(kServiceName), service_main},
        {nullptr, nullptr}
    };
    if (!::StartServiceCtrlDispatcherA(table)) { if (err) *err = "StartServiceCtrlDispatcher"; return false; }
    return true;
}
#endif

}  // namespace nova::system
