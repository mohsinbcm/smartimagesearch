# Git Flow Strategy

This document outlines the Git workflow and branching strategy for the Smart Image Search project.

## Overview

We use a **modified Git Flow** that allows **selective feature promotion** to releases:

**Key Principle**: 
- **develop** = Integration branch with ALL completed features (released + unreleased)
- **release/** = Created from `main`, selectively includes features
- **main** = Production releases only
- **main → develop** = After release, sync develop with production

This approach allows us to have multiple features in `develop` but choose which ones go into each release.

## Branch Strategy

```
main (production, tagged releases)
  ↑
  ├── release/v0.2.0 (selective features)
  │   ↑
  │   ├── feature/vector-search (merged to release)
  │   └── feature/face-recognition (merged to release)
  │
  └── develop (all features, integration)
      ├── feature/vector-search (also merged here)
      ├── feature/face-recognition (also merged here)
      └── feature/video-indexing (only here, not in release yet)
```

**Flow**:
1. Features branch from `develop`
2. Release branches from `main` (not `develop`)
3. Release selectively merges features
4. Release merges to `main`
5. `main` merges back to `develop` (keeps develop in sync)
6. Unreleased features remain in `develop` for future releases

## Branch Types

### Main Branches

#### `main`
- Always production-ready code
- Protected branch (no direct commits)
- Tagged with version numbers (v0.1.0, v0.2.0, etc.)
- Only receives merges from `release/*` or `hotfix/*` branches
- **Never** merge directly from feature or develop branches

#### `develop`
- Integration and testing branch
- Contains ALL completed features (released and unreleased)
- Base branch for all feature branches
- Receives syncs from `main` after each release
- Features can be merged here anytime when complete
- May be ahead of `main` with unreleased features

### Supporting Branches

#### Feature Branches (always) AND `release/*` (when ready for release)
- **Purpose**: Develop new features
- **Lifecycle**: Can merge to `develop` when complete, even if not ready for release}`
- **Branch from**: `develop`
- **Merge into**: `develop`
- **Purpose**: Develop new features
- **Examples**:
  - `feature/image-crawler`
  - `feature/clip-embeddings`
  - `feature/face-recognition`
  - `feature/vector-search`

#### Bugfix Branches
- **Naming**: `bugfix/{issue-description}`
- **Branch from**: `develop`
- **Merge into**: `develop`
- **Purpose**: Fix non-critical bugs
- **Examples**:
  - `bugfix/fix-duplicate-detection`
  - `bugfix/memory-leak-embeddings`

#### Hotfix Branches
- **Naming**: `hotfix/{critical-fix}`
- **Branch from**: `main`
- **Merge into**: `main` AND `develop`
- **Purpose**: Urgent production fixes
- **Examples**:
  - `hotfix/critical-security-patch`
  - `hotfix/database-corruption`

#### Experiment Branches
- **Naming**: `experiment/{test-name}`
- **Branch from**: `develop`
- **Merge into**: `develop` (if successful)
- **Purpose**: ML experiments, model testing
- **Examples**:
  - `experiment/blip2-embeddings`
  - `experiment/hnsw-index`

#### Refactor Branches
- **Naming**: `refactor/{component}`
- **Branch from**: `develop`
- **Merge into**: `develop`
- **Purpose**: Code restructuring without changing functionality
- **Examples**:
  - `refactor/database-layer`
  - `refactor/api-endpoints`

#### Documentation Branches
- **Naming**: `docs/{documentation}`
- **Branch from**: `develop`
- **Merge into**: `develop`
- **Purpose**: Documentation updates
- **Examples**:
  - `docs/api-reference`
  - `docs/installation-guide`

## Commit Message Convention

We follow [Conventional Commits](https://www.conventionalcommits.org/) specification.

### Format
```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Types

- **feat**: New feature
  ```bash
  feat(search): add vector similarity search with FAISS
  ```

- **fix**: Bug fix
  ```bash
  fix(crawler): resolve memory leak in batch processing
  ```

- **perf**: Performance improvement
  ```bash
  perf(face): optimize face detection speed by 30%
  ```

- **docs**: Documentation changes
  ```bash
  docs(readme): update installation instructions
  ```

- **test**: Adding or updating tests
  ```bash
  test(embeddings): add unit tests for CLIP encoder
  ```

- **refactor**: Code restructuring
  ```bash
  refactor(db): reorganize database schema
  ```

- **chore**: Maintenance tasks
  ```bash
  chore(deps): update transformers to 4.35.0
  ```

- **style**: Code formatting (no functional change)
  ```bash
  style: format code with black
  ```

- **build**: Build system or dependencies
  ```bash
  build: update pyproject.toml dependencies
  ```

- **ci**: CI/CD configuration
  ```bash
  ci: add GitHub Actions workflow
  ```

### Scope Examples

- `search`: Search functionality
- `face`: Face recognition
- `crawler`: Image crawler
- `db`: Database
- `api`: API endpoints
- `ui`: User interface
- `embeddings`: Embedding generation

### Commit Message Examples

```bash
# Feature
feat(face): implement face clustering with DBSCAN
feat(api): add person search endpoint

# Bug fix
fix(search): correct similarity threshold calculation
fix(crawler): handle permission errors gracefully

# Performance
perf(embeddings): reduce batch processing memory usage
perf(db): add index on image_path column

# Documentation
docs(contributing): add code review guidelines
docs(api): document search endpoints

# Breaking change
feat(api)!: change search response format

BREAKING CHANGE: Search API now returns results as array instead of object
```

## Workflow

### 1. Starting a New Feature

```bash
# Update develop branch
git checkout develop
git pull origin develop

# Create feature branch
git checkout -b feature/image-crawler

# Work on feature
git add .
git commit -m "feat(crawler): implement recursive directory scanning"

# Push to remote
git push -u origin feature/image-crawler
```

### 2. Working on Feature

```bash
# Make changes
git add .
git commit -m "feat(crawler): add EXIF metadata extraction"

# Push changes
git push
```

### 3. Keeping Feature Branch Updated

```bash
# Fetch latest changes
git fetch origin

# Rebase on develop (preferred)
git rebase origin/develop

# Or merge if rebase is complex
git merge origin/develop

# Push (force push if rebas (Merge to Develop)

```bash
# Ensure branch is up to date
git fetch origin
git rebase origin/develop

# Push final changes
git push

# Merge to develop (feature complete but may not be released yet)
git checkout develop
git merge --no-ff feature/image-crawler -m "feat: integrate image crawler"
git push origin develop

# Keep feature branch until it's in a release
# Don't delete yet - we might need it for the release branch
```

### 5. Creating a Release (Selective Feature Promotion)

This is the key difference: release branches come from `main`, not `develop`.

```bash
# === STEP 1: Create release branch from main ===
git checkout main
git pull origin main
git checkout -b release/v0.2.0

# === STEP 2: Selectively merge ready features ===
# Choose which features to include in this release

# Include vector search (ready)
git merge --no-ff feature/vector-search -m "feat: add vector search for v0.2.0"

# Include face recognition (ready)
git merge --no-ff feature/face-recognition -m "feat: add face recognition for v0.2.0"

# DON'T merge video-indexing (not ready yet)
# It stays in develop for the next release

# === STEP 3: Prepare release ===
# Update version in pyproject.toml
sed -i 's/version = ".*"/version = "0.2.0"/' pyproject.toml

# Update CHANGELOG.md
cat << 'EOF' >> CHANGELOG.md

## [0.2.0] - 2025-12-31

### Added
- Vector similarity search with FAISS
- Face detection and recognition
- Person directory management

### 7. Creating a Hotfix

Hotfixes follow the standard flow (unchanged):

```bash
# Create hotfix branch from main
git checkout main
git checkout -b hotfix/security-patch

# Fix the issue
git commit -am "fix(security): patch XSS vulnerability"

# Merge to main
git checkout main
git merge --no-ff hotfix/security-patch
git tag -a v0.2.1 -m "Hotfix v0.2.1: Security patch"
git push origin main --tags

# Merge to develop to keep it in sync
git checkout develop
git merge --no-ff hotfix/security-patch
git push origin develop

# Delete hotfix branch
git branch -d hotfix/security-patch
git push origin --delete hotfix/security-patch
```
 → Release v0.1.0
```bash
# Develop features
git checkout develop
git checkout -b feature/image-crawler
git checkout -b feature/metadata-extraction
git checkout -b feature/file-watcher

# Merge to develop when complete
git checkout develop
git merge --no-ff feature/image-crawler
git merge --no-ff feature/metadata-extraction
git merge --no-ff feature/file-watcher

# Create release v0.1.0 from main
git checkout main
git checkout -b release/v0.1.0
git merge --no-ff feature/image-crawler
git merge --no-ff feature/metadata-extraction
git merge --no-ff feature/file-watcher

# Release and sync
git checkout main
git merge --no-ff release/v0.1.0
git tag -a v0.1.0 -m "v0.1.0: Image discovery and indexing"
git checkout develop
git merge --no-ff main
```

### Phase 2: Embeddings (Week 2) → Release v0.2.0
```bash
# Develop features
git checkout develop
git checkout -b feature/clip-embeddings
git checkout -b feature/batch-processing
git checkout -b feature/thumbnail-generation

# Merge to develop
git checkout develop
git merge --no-ff feature/clip-embeddings
git merge --no-ff feature/batch-processing
git merge --no-ff feature/thumbnail-generation

# Release v0.2.0 with embeddings
git checkout main
git checkout -b release/v0.2.0
git merge --no-ff feature/clip-embeddings
git merge --no-ff feature/batch-processing
git merge --no-ff feature/thumbnail-generation
# ... release process
```

### Phase 2.5: Face Recognition (Week 2-3) → Release v0.3.0 or v0.2.5
```bash
git checkout develop
git checkout -b feature/face-detection
git checkout -b feature/face-clustering
git checkout -b feature/person-directory

# These can be released separately or together
# Flexible based on completion and testing
```

### Phase 3-6: Iterative Releases
```bash
# Develop multiple features in parallel
# Merge to develop as they complete
# Selectively promote to releases based on:
#   - Feature completeness
#   - Testing status
#   - Business priorities
#   - Breaking changes
# Step 7: Cleanup
git branch -d release/v0.2.0
git branch -d feature/vector-search
git branch -d feature/face-recognition

# === Result ===
# main: v0.2.0 with vector-search + face-recognition
# develop: v0.2.0 + video-indexing (ready for v0.3.0)

# === Next Release v0.3.0 ===
git checkout main
git checkout -b release/v0.3.0

# Now include video indexing
git merge --no-ff feature/video-indexing

# ... proceed with release
- Person directory and clustering
- 3x faster GPU indexing

Performance:
- Optimized batch processing
- GPU acceleration support

Breaking Changes: None"

# === STEP 7: Push main and tags ===
git push origin main --tags

# === STEP 8: Sync develop with released features ===
# This is critical - brings released code back to develop
git checkout develop
git pull origin develop
git merge --no-ff main -m "chore: sync develop with v0.2.0 release"

# Resolve any conflicts if they occur
# Usually none if workflow is followed correctly

git push origin develop

# === STEP 9: Merge unreleased features to develop (if not already) ===
# If feature/video-indexing wasn't in develop yet
git merge --no-ff feature/video-indexing -m "feat: integrate video indexing (for v0.3.0)"
git push origin develop

# === STEP 10: Cleanup ===
git branch -d release/v0.2.0
git push origin --delete release/v0.2.0

# Delete merged feature branches
git branch -d feature/vector-search
git branch -d feature/face-recognition
git push origin --delete feature/vector-search
git push origin --delete feature/face-recognition

# Keep feature/video-indexing if still working on it
```

### 6. Release Decision Matrix

Before creating a release, decide which features to include:

```bash
# List all feature branches
git branch -r | grep feature/

# Check what's in develop but not in main
git log main..develop --oneline --graph

# For each feature, ask:
# ✅ Is it complete?
# ✅ Is it tested?
# ✅ Is it documented?
# ✅ Does it have breaking changes?
# ✅ Is it ready for users?

# Then selectively merge to release branch
git push origin main develop --tags

# Delete release branch
git branch -d release/v0.1.0
```

### 6. Creating a Hotfix

```bash
# Create hotfix branch from main
git checkout -b hotfix/security-patch main

# Fix the issue
git commit -am "fix(security): patch XSS vulnerability"

# Merge to main
git checkout main
git merge --no-ff hotfix/security-patch
git tag -a v0.1.1 -m "Hotfix v0.1.1: Security patch"

# Merge to develop
git checkout develop
git merge --no-ff hotfix/security-patch

# Push
git push origin main develop --tags

# Delete hotfix branch
git branch -d hotfix/security-patch
```

## Project Phases Implementation Plan

### Phase 1: Image Discovery (Week 1)
```bash
git checkout -b feature/image-crawler develop
git checkout -b feature/metadata-extraction develop
gitWhy This Approach?

### Advantages ✅
1. **Selective Feature Promotion**: Choose exactly which features go into each release
2. **Clean Production**: Main only contains released, tested features
3. **Flexible Development**: Features can be completed and merged to develop anytime
4. **No Long-Lived Branches**: Features don't sit in branches for weeks
5. **Clear Release Scope**: Release branch shows exactly what's being released
6. **Safe Experimentation**: Experimental features can live in develop without affecting releases
7. **Easy Rollback**: Can exclude problematic features from release easily

### When to Use Each Branch
- **Use `develop`**: Daily development, integration testing, experiments
- **Use `release/*`**: When preparing a specific version for production
- **Use `main`**: When you need production-ready code
- **Use `feature/*`**: When working on isolated functionality

## Best Practices

### DO ✅
- **Always** create release branches from `main`, not `develop`
- **Always** sync `develop` from `main` after a release
- Write clear, descriptive commit messages
- Keep commits atomic (one logical change per commit)
- Rebase feature branches regularly on `develop`
- Delete merged feature branches after release
- Tag all releases with semantic versioning
- Update CHANGELOG.md for releases with "Not Included" section
- Run tests before merging to release branch
- Document which features are included/excluded in each release

### DON'T ❌
- **Never** merge `develop` directly to `main`
- **Never** merge `develop` to `release/*` branch
- **Never** create release branch from `develop`
- Commit directly to main or develop
- Commit large binary files (use Git LFS)
- Mix unrelated changes in one commit
- Force push to main or develop
- Leave dead branches around
- Commit sensitive data (API keys, passwords)
- Commit generated files (models, cache, logs)
- Forget to sync develop from main after release
git checkout -b feature/faiss-index develop
git checkout -b feature/vector-search develop
git checkout -b feature/query-processing develop
```

### Phase 4: API Backend (Week 4)
```Common Scenarios

### Scenario 1: Feature Ready Mid-Sprint
```bash
# Feature completed but release not ready yet
git checkout develop
git merge --no-ff feature/new-feature

# Feature stays in develop until next release
# It will be selectively picked for appropriate release
```

### Scenario 2: Feature Not Ready for Release
```bash
# Feature in develop but not stable enough
# Simply don't merge it to release branch

git checkout main
git checkout -b release/v0.3.0
# Only merge stable features
git merge --no-ff feature/stable-feature
# Skip feature/unstable-feature
```

### Scenario 3: Need to Exclude Feature After Merge to Develop
```bash
# Feature merged to develop but has critical bug
# Don't merge it to release branch
# Fix it in develop for next release

git checkout develop
git checkout -b bugfix/fix-feature
# Fix the bug
git checkout develop
git merge --no-ff bugfix/fix-feature

# Next release can include the fixed feature
```

### Scenario 4: Sync Conflicts (Develop vs Main)
```bash
# After releasing, develop sync might have conflicts
git checkout develop
git merge --no-ff main
# CONFLICT in some files

# This happens when develop has unreleased changes
# Resolve by keeping develop's version (unreleased features)

git status
# Edit conflicted files
git add .
git commit -m "chore: resolve sync conflicts, keep develop changes"
```

### Scenario 5: Multiple Features with Dependencies
```bash
# Feature B depends on Feature A
# Both must go in same release

git checkout main
git checkout -b release/v0.4.0

# Merge in order
git merge --no-ff feature/feature-a
git merge --no-ff feature/feature-b

# Test together before releasing
```

## Troubleshooting

### "I merged to develop but now I don't want it in the release"
```bash
# No problem! Just don't merge that feature to release branch
# It will stay in develop until next release
```

### "I forgot to sync develop after release"
```bash
# Sync now
git checkout develop
git pull origin develop
git merge --no-ff main
git push origin develop
```

### "I accidentally merged develop to release"
```bash
# Reset release branch
git checkout release/v0.x.0
git reset --hard main

# Start over, merge only intended features
git merge --no-ff feature/intended-feature
```

### "Feature branch has conflicts with release branch"
```bash
# Option 1: Update feature branch first
git checkout feature/my-feature
git merge main
# Resolve conflicts
git checkout release/v0.x.0
git merge --no-ff feature/my-feature

# Option 2: Resolve during release merge
git checkout release/v0.x.0
git merge --no-ff feature/my-feature
# Resolve conflicts here
```

### Undo last commit (keep changes)
```bash
git reset --soft HEAD~1
```

### Undo last commit (discard changes)
```bash
git reset --hard HEAD~1
```

### Recover deleted branch
```bash
git reflog
git checkout -b feature/recovered <commit-hash>
```

### See what's in develop but not in main
```bash
git log main..develop --oneline
git diff main..develop --stat

#### For `main` branch:
```yaml
Settings → Branches → Branch protection rules

Rules:
  - Require pull request reviews before merging: 1 approval
  - Require status checks to pass before merging: Yes
    - Required checks: pytest, code-quality, type-check
  - Require branches to be up to date before merging: Yes
  - Require linear history: Yes
  - Include administrators: No
  - Allow force pushes: No
  - Allow deletions: No
```

#### For `develop` branch:
```yaml
Settings → Branches → Branch protection rules

Rules:
  - Require pull request reviews before merging: No (for solo)
  - Require status checks to pass before merging: Yes
    - Required checks: pytest, code-quality
  - Require branches to be up to date before merging: No
  - Include administrators: No
  - Allow force pushes: No
  - Allow deletions: No
```

### Pull Request Template

Create `.github/pull_request_template.md`:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] New feature
- [ ] Bug fix
- [ ] Performance improvement
- [ ] Refactoring
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Manual testing completed
- [ ] Performance impact assessed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
- [ ] All tests pass
```

## Versioning

We follow [Semantic Versioning](https://semver.org/):

```
MAJOR.MINOR.PATCH

Example: v1.2.3
  1 = Major version (breaking changes)
  2 = Minor version (new features, backward compatible)
  3 = Patch version (bug fixes)
```

### Version Increments

- **Major (v1.0.0 → v2.0.0)**: Breaking API changes, major architecture changes
- **Minor (v1.0.0 → v1.1.0)**: New features, new models, backward compatible
- **Patch (v1.0.0 → v1.0.1)**: Bug fixes, performance improvements, documentation

### Pre-release Tags

```bash
v0.1.0-alpha.1  # Alpha release
v0.1.0-beta.1   # Beta release
v0.1.0-rc.1     # Release candidate
```

## Quick Reference Commands

```bash
# Start new feature
git checkout develop && git pull
git checkout -b feature/vector-search

# Regular work
git add .
git commit -m "feat(search): implement FAISS vector search"
git push -u origin feature/vector-search

# Update from develop
git fetch origin
git rebase origin/develop
git push --force-with-lease

# Merge feature (after PR)
git checkout develop
git merge --no-ff feature/vector-search
git push origin develop
git branch -d feature/vector-search

# Create release
git checkout -b release/v1.0.0 develop
# ... update version and CHANGELOG ...
git checkout main
git merge --no-ff release/v1.0.0
git tag -a v1.0.0 -m "Version 1.0.0"
git checkout develop
git merge --no-ff release/v1.0.0
git push origin main develop --tags

# View branch graph
git log --oneline --graph --all --decorate

# Check current branch
git branch --show-current

# List all branches
git branch -a

# Delete merged branches
git branch --merged develop | grep -v "^\*\|main\|develop" | xargs git branch -d
```

## Best Practices

### DO ✅
- Write clear, descriptive commit messages
- Keep commits atomic (one logical change per commit)
- Rebase feature branches regularly
- Delete merged branches
- Tag all releases
- Update CHANGELOG.md for releases
- Run tests before pushing
- Review your own code before creating PR

### DON'T ❌
- Commit directly to main
- Commit large binary files (use Git LFS)
- Mix unrelated changes in one commit
- Force push to main or develop
- Leave dead branches around
- Commit sensitive data (API keys, passwords)
- Commit generated files (models, cache, logs)

## Useful Aliases

Add to `~/.gitconfig`:

```bash
[alias]
  # Short status
  st = status -sb
  
  # Pretty log
  lg = log --graph --pretty=format:'%Cred%h%Creset -%C(yellow)%d%Creset %s %Cgreen(%cr) %C(bold blue)<%an>%Creset' --abbrev-commit
  
  # Show branches
  br = branch -vv
  
  # Quick commit
  cm = commit -m
  
  # Amend last commit
  amend = commit --amend --no-edit
  
  # Update from remote
  up = pull --rebase --autostash
  
  # Clean merged branches
  cleanup = "!git branch --merged | grep -v '\\*\\|main\\|develop' | xargs -n 1 git branch -d"
```

## Troubleshooting

### Undo last commit (keep changes)
```bash
git reset --soft HEAD~1
```

### Undo last commit (discard changes)
```bash
git reset --hard HEAD~1
```

### Resolve merge conflicts
```bash
git fetch origin
git merge origin/develop
# ... fix conflicts ...
git add .
git commit
```

### Recover deleted branch
```bash
git reflog
git checkout -b feature/recovered <commit-hash>
```

---

**Last Updated**: December 31, 2025
