#!/usr/bin/env bash
set -euo pipefail

ISAAC_ROOT="${ISAAC_ROOT:-D:\\isaacsim}"
DISTRO_NAME="${WSL_DISTRO_NAME:-Ubuntu-24.04}"
REPO_UNC="\\\\wsl.localhost\\${DISTRO_NAME}$(pwd | sed 's#/#\\#g')"
SCRIPT_UNC="${REPO_UNC}\\scripts\\render_isaac_camera_windows.py"

to_unc_path() {
  local path="$1"
  local abs_path
  if [[ "${path}" = /* ]]; then
    abs_path="${path}"
  else
    abs_path="$(pwd)/${path}"
  fi
  printf '\\\\wsl.localhost\\%s%s' "${DISTRO_NAME}" "$(printf '%s' "${abs_path}" | sed 's#/#\\#g')"
}

quote_ps_arg() {
  local value="$1"
  value="${value//\'/\'\'}"
  printf "'%s'" "${value}"
}

win_args=()
expect_dataf=false
for arg in "$@"; do
  if [[ "${expect_dataf}" == true ]]; then
    win_args+=("$(to_unc_path "${arg}")")
    expect_dataf=false
    continue
  fi
  case "${arg}" in
    --dataf)
      win_args+=("${arg}")
      expect_dataf=true
      ;;
    --dataf=*)
      win_args+=("--dataf=$(to_unc_path "${arg#--dataf=}")")
      ;;
    *)
      win_args+=("${arg}")
      ;;
  esac
done

cmdline="& $(quote_ps_arg "${ISAAC_ROOT}\\python.bat") $(quote_ps_arg "${SCRIPT_UNC}")"
for arg in "${win_args[@]}"; do
  cmdline+=" $(quote_ps_arg "${arg}")"
done

"/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe" -NoProfile -ExecutionPolicy Bypass -Command "${cmdline}"
