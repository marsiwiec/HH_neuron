{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

{
  # https://devenv.sh/languages/
  languages.python = {
    enable = true;
    uv.enable = true;
  };
}
