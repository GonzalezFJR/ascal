# Public web app

The service at <https://lumaria.allandestars.com/ascal/> is `ascal.web.app` (the same app as `ascal web`) running in
Docker on a machine with enough CPU and memory, reached by the public web server through a reverse SSH tunnel:

```
browser ──https──> nginx (EC2, lumaria.allandestars.com) ── /ascal/ ──> 127.0.0.1:8790 (EC2)
                                                                            │ reverse SSH tunnel
                                                                            ▼
                                                    compute host: 127.0.0.1:8790 -> container ascal-web:8000
```

## Compute host

```bash
rsync -az --delete --exclude .git --exclude notebooks ./ host:~/ascal-web/     # from the repository root
ssh host 'cd ~/ascal-web/deploy && docker compose up -d --build'
```

Limits are set in `docker-compose.yml` (upload size, jobs per IP and hour, queue length, how long results are kept,
largest time budget; see the docstring of `ascal/web/app.py`). One job runs at a time, each in its own process; a
77-Mpx FITS frame needs about 2.6 GB and 65 s on a 4-core machine. Uploaded images are deleted as soon as the job
ends; results after `ASCAL_WEB_TTL_H` hours. Examples offered on the page are listed in `examples/web/examples.json`.

## Tunnel

A dedicated key that can only open the listening port 127.0.0.1:8790 on the web server:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/ascal_tunnel -N "" -C ascal-tunnel@host
# on the web server, in ~/.ssh/authorized_keys:
restrict,port-forwarding,permitlisten="127.0.0.1:8790",command="/bin/false" ssh-ed25519 AAAA... ascal-tunnel@host
```

Then install `ascal-tunnel.service` as a user unit on the compute host (`~/.config/systemd/user/`),
`systemctl --user enable --now ascal-tunnel` and `loginctl enable-linger $USER`.

## Web server

Add `nginx-ascal.conf` to the `server` block of the site and reload nginx. The app uses relative URLs only, so the
prefix is stripped by `proxy_pass http://127.0.0.1:8790/;`.
