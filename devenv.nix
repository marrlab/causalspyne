{ pkgs, lib, config, inputs, ... }:
let
  pkgs-unstable = import inputs.nixpkgs-unstable { system = pkgs.stdenv.system; };
in
{
  # https://devenv.sh/packages/
  packages = [ pkgs.git ];

  # https://devenv.sh/languages/
  languages.python = {
    enable = true;
    version = "3.12";
    uv = {
      enable = true;
      package = pkgs-unstable.uv;
      sync.enable = true;
    };
    venv.enable = true;
  };

  enterShell = ''
  '';

  # See full reference at https://devenv.sh/reference/options/
}
