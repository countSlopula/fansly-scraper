{
  description = "Dev shell for Jakan-Kink/fansly-scraper on NixOS (project-local PostgreSQL)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};

      # tkinter is only used for the interactive directory picker on first run.
      # The Poetry venv sees it via system-site-packages (set below).
      python = pkgs.python312.withPackages (ps: with ps; [ tkinter ]);

      # Shared libraries the manylinux wheels dlopen() at import time.
      #
      # NOTE: nix-ld does NOT help here. It fixes the ELF interpreter for
      # foreign *executables*; CPython is a Nix binary and resolves its
      # extension modules through the normal dynamic linker, so
      # LD_LIBRARY_PATH is the correct mechanism.
      wheelLibs = with pkgs; [
        stdenv.cc.cc.lib # libstdc++ -> plyvel-ci (leveldb), numpy/scipy
        zlib # pillow, imagehash
        openssl # psycopg / asyncpg
        libffi
      ];

      postgres = pkgs.postgresql_17;

      # 5432: the default the test fixtures resolve to (FANSLY_PG_PORT), so
      # `pytest` needs no extra env. There is no system postgres on NixOS here
      # to collide with; if one is ever enabled, override FANSLY_PG_PORT.
      pgPort = "5432";
      pgName = "fansly"; # used for both the superuser role and the database

      # postgres has its own openssl/zlib in its RPATH; keep the wheel
      # LD_LIBRARY_PATH out of its environment so nothing gets shadowed.
      pgEnv = ''unset LD_LIBRARY_PATH'';

      pg-start = pkgs.writeShellScriptBin "pg-start" ''
        set -euo pipefail
        ${pgEnv}

        mkdir -p "$PGHOST"

        if [ ! -d "$PGDATA" ]; then
          echo "initialising cluster in $PGDATA ..."
          ${postgres}/bin/initdb -D "$PGDATA" -U "$PGUSER" \
            --no-locale --encoding=UTF8 >/dev/null
        fi

        if ${postgres}/bin/pg_ctl -D "$PGDATA" status >/dev/null 2>&1; then
          echo "already running on port $PGPORT"
        else
          ${postgres}/bin/pg_ctl -D "$PGDATA" -l "$PGDATA/server.log" -w start \
            -o "-k $PGHOST -h 127.0.0.1 -p $PGPORT"
        fi

        if ! ${postgres}/bin/psql -tAc \
             "SELECT 1 FROM pg_database WHERE datname='$PGDATABASE'" | grep -q 1; then
          ${postgres}/bin/createdb "$PGDATABASE"
          echo "created database $PGDATABASE"
        fi
      '';

      pg-stop = pkgs.writeShellScriptBin "pg-stop" ''
        set -euo pipefail
        ${pgEnv}
        ${postgres}/bin/pg_ctl -D "$PGDATA" -m fast stop
      '';

      pg-destroy = pkgs.writeShellScriptBin "pg-destroy" ''
        set -euo pipefail
        ${pgEnv}
        ${postgres}/bin/pg_ctl -D "$PGDATA" -m immediate stop 2>/dev/null || true
        rm -rf "$PGDATA" "$PGHOST"
        echo "cluster removed"
      '';
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        packages = [
          python
          pkgs.poetry
          pkgs.nodejs_22 # JSPyBridge spawns this at RUNTIME; match .nvmrc
          pkgs.ffmpeg # ffmpeg-python shells out to the binary
          postgres # server + psql/createdb
          pg-start
          pg-stop
          pg-destroy
        ];

        env = {
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath wheelLibs;
          POETRY_VIRTUALENVS_IN_PROJECT = "true";
          POETRY_VIRTUALENVS_OPTIONS_SYSTEM_SITE_PACKAGES = "true";

          PGPORT = pgPort;
          PGUSER = pgName;
          PGDATABASE = pgName;

          # The test fixtures resolve their own PG connection params and default
          # the user to $USER, which is not a role in this cluster. Point them at
          # the cluster's superuser so `pytest` needs no per-invocation env.
          FANSLY_PG_HOST = "127.0.0.1";
          FANSLY_PG_PORT = pgPort;
          FANSLY_PG_USER = pgName;
        };

        shellHook = ''
          # Must be set here rather than in `env`: they depend on $PWD.
          export PGDATA="$PWD/.pgdata"
          export PGHOST="$PWD/.pgsock"

          export FANSLY_PYTHON=${python}/bin/python

          cat <<'EOF'
        First-time setup:
          poetry env use "$FANSLY_PYTHON"
          poetry install --no-root --with browser-auth
          npm install acorn acorn-walk

        Then, each session:
          pg-start
          poetry run python fansly_downloader_ng.py
          pg-stop
        EOF
        '';
      };
    };
}
