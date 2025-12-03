# sync.sh - À la racine du projet
#!/bin/bash

cd "$(dirname "$0")"

echo "🔄 Synchronisation Git..."

# Pull les derniers changements
git fetch origin
git pull origin main

# Afficher le statut
git status

echo "✅ Sync terminé"
echo "📂 Modèles partagés: /mnt/shared/ploutos_data/models/"
ls -lh /mnt/shared/ploutos_data/models/ 2>/dev/null || echo "⚠️  NFS non monté"
